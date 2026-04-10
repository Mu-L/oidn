// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sycl_conv.h"
#include "sycl_common.h"

OIDN_NAMESPACE_BEGIN
OIDN_ARCH_XE_NAMESPACE_BEGIN

  template<typename SrcDstT, typename WeightT, TensorLayout srcDstLayout, TensorLayout weightLayout>
  struct SYCLConvBaseKernel
  {
    static constexpr int KW = 3;        // kernel width
    static constexpr int KH = 3;        // kernel height
    static constexpr int PW = (KW-1)/2; // padding width on each side
    static constexpr int PH = (KH-1)/2; // padding height on each side

    static constexpr int dpasDepth  = 8; // DPAS depth
    static constexpr int dpasRepeat = 8; // DPAS repeat count

    static constexpr int blockC = TensorByteOffset<SrcDstT, srcDstLayout>::blockC; // channel block size

  #if defined(OIDN_ARCH_XEHPG)
    using MatmulT = SrcDstT;
    static constexpr int blockOC = 8;                 // output channel block size (= exec width)
    static constexpr int blockOCB = blockC / blockOC; // block of output channel blocks
    static constexpr int blockOH = 6; // block output height
  #elif defined(OIDN_ARCH_XEHPC)
    using MatmulT = SrcDstT;
    static constexpr int blockOH = 4; // output block height
  #elif defined(OIDN_ARCH_XE2)
    using MatmulT = SrcDstT;
    static constexpr int blockOH = 6; // output block height
  #else
    using MatmulT = float; // no DPAS -> use FP32 FMAs
    static constexpr int blockOH = 2; // output block height
  #endif

    static constexpr int blockOW = dpasRepeat;       // output block width
    static constexpr int blockIW = blockOW + KW - 1; // input block width

    static_assert(blockOH % 2 == 0, "blockOH must be even");
    static_assert(blockOW % 2 == 0, "blockOW must be even");

    #if defined(OIDN_ARCH_XEHPG)
      // FP32 accumulator rows
      using AccumRows = simd<float, blockOW * blockOC>[blockOH][blockOCB];
    #else
      // FP32 accumulator rows
      using AccumRows = simd<float, blockOW * blockC>[blockOH];
    #endif

    using InRows  = simd<MatmulT, blockIW * blockC>[blockOH];
    using OutRows = simd<SrcDstT, blockOW * blockC>[blockOH];

    // Loads a row from a src tensor, assuming the row is within the tensor bounds
    template<int N>
    static oidn_inline void loadInnerRow(simd<MatmulT, N>& row,
                                         const TensorAccessor3D<SrcDstT, srcDstLayout>& src,
                                         int ic, int ih, int iw)
    {
      static_assert(N % blockC == 0, "non-integer width");
      constexpr int W = N / blockC;

      if (iw >= 0 && iw + W <= src.W)
      {
        // Fast path: load the entire row
        const SrcDstT* srcPtr = &src(ic, ih, iw);
        row = loadLargeBlock<SrcDstT, N>(srcPtr);
      }
      else
      {
        // Slow path: load the in-bounds pixels of the row
        const simd<int, W> iwVec(iw, 1); // iw, iw+1, iw+2, ...
        simd_mask<W> predVec = (iwVec >= 0) & (iwVec < src.W);
        simd<uint32_t,  W> srcOffsetVec = src.getByteOffset(ic, ih, 0) +
                                          iwVec * src.getByteOffset.wByteStride;
        simd<uintptr_t, W> srcAddrVec   = reinterpret_cast<uintptr_t>(src.ptr) + srcOffsetVec;

        #pragma unroll
        for (int w = 0; w < W; ++w)
        {
          const SrcDstT* srcPtr = reinterpret_cast<const SrcDstT*>(uintptr_t(srcAddrVec[w]));
          row.template select<blockC, 1>(w * blockC) =
            loadBlock<SrcDstT, blockC>(srcPtr, predVec.template select<1, 1>(w));
        }
      }
    }

    template<int N>
    static oidn_inline void loadRow(simd<MatmulT, N>& row,
                                    const TensorAccessor3D<SrcDstT, srcDstLayout>& src,
                                    int ic, int ih, int iw)
    {
      if (ih < 0 || ih >= src.H)
      {
        row = 0;
        return;
      }

      loadInnerRow(row, src, ic, ih, iw);
    }

    // Loads a row from a virtually 2x upsampled src tensor
    template<int N>
    static oidn_inline void loadUpsampleRow(simd<MatmulT, N>& row,
                                            const TensorAccessor3D<SrcDstT, srcDstLayout>& src,
                                            int ic, int ih, int iw)
    {
      static_assert(N % blockC == 0, "non-integer width");
      constexpr int W = N / blockC;
      static_assert(W % 2 == 0, "width must be even for 2x upscale");

      ih >>= 1;
      iw >>= 1;

      if (ih < 0 || ih >= src.H)
      {
        row = 0;
        return;
      }

      constexpr int alignW = PW & 1; // assumption: iw%2 == PW%2
      constexpr int srcW = (W >> 1) + alignW; // number of source pixels to load

      simd<MatmulT, srcW * blockC> srcRow;
      loadInnerRow(srcRow, src, ic, ih, iw);

      // Upsample by copying the pixels from the source row to the destination row
      for (int i = 0; i < W; ++i)
      {
        row.template select<blockC, 1>(i * blockC) =
          srcRow.template select<blockC, 1>(((i + alignW) >> 1) * blockC);
      }
    }

    // Stores a row to the dst tensor
    // Pixels can be stored in chunks of K to improve performance
    template<int K = 1, int N>
    static oidn_inline void storeRow(simd<SrcDstT, N>& row,
                                     const TensorAccessor3D<SrcDstT, srcDstLayout>& dst,
                                     int oc, int oh, int ow)
    {
      static_assert(N % blockC == 0, "non-integer width");
      constexpr int W = N / blockC;
      static_assert(W % K == 0, "non-integer chunks");

      //if (oh >= dst.H)
      //  return;

      if (ow + W <= dst.W)
      {
        // Fast path: store the entire row
        SrcDstT* dstPtr = &dst(oc, oh, ow);
        storeLargeBlock(dstPtr, row);
      }
      else
      {
        // Slow path: store the in-bounds pixels of the row
        constexpr int numChunks = W / K;
        constexpr int chunkSize = blockC * K;

        const simd<int, numChunks> owVec(ow, K); // ow, ow+K, ow+2*K, ...
        simd_mask<numChunks> predVec = owVec < dst.W;
        simd<uint32_t,  numChunks> dstOffsetVec = dst.getByteOffset(oc, oh, 0) +
                                                  owVec * dst.getByteOffset.wByteStride;
        simd<uintptr_t, numChunks> dstAddrVec   = reinterpret_cast<uintptr_t>(dst.ptr) + dstOffsetVec;

        #pragma unroll
        for (int i = 0; i < numChunks; ++i)
        {
          SrcDstT* dstPtr = reinterpret_cast<SrcDstT*>(uintptr_t(dstAddrVec[i]));
          storeBlock(dstPtr, row.template select<chunkSize, 1>(i * chunkSize).read(),
                     predVec.template select<1, 1>(i));
        }
      }
    }

    // Multiply + accumulate block for one kernel row (kh)
    static oidn_inline void mmaBlockKW(AccumRows& accumRows, InRows& inRows, const WeightT* weightPtr, int kh)
    {
      // Iterate over kernel width
      #pragma unroll
      for (int kw = 0; kw < KW; ++kw)
      {
        // Load weight matrix for kernel tap
      #if defined(OIDN_ARCH_XEHPG)
        simd<MatmulT, blockOC * blockC> weightMat[blockOCB];
        #pragma unroll
        for (int bocb = 0; bocb < blockOCB; ++bocb)
        {
          weightMat[bocb] = loadBlock<WeightT, blockOC * blockC>(weightPtr);
          weightPtr += blockOC * blockC;
        }
      #else
        simd<MatmulT, blockC * blockC> weightMat = loadLargeBlock<WeightT, blockC * blockC>(weightPtr);
        weightPtr += blockC * blockC;
      #endif

        // Multiply + accumulate rows
      #if defined(OIDN_ARCH_XEHPG)
        #pragma unroll
        for (int boh = 0; boh < blockOH; ++boh)
        {
          #pragma unroll
          for (int bocb = 0; bocb < blockOCB; ++bocb)
          {
            accumRows[boh][bocb] = xmx::dpas<dpasDepth, dpasRepeat, float>(
              accumRows[boh][bocb],
              weightMat[bocb],
              inRows[(kh + boh) % blockOH].template select<blockOW * blockC, 1>(kw * blockC).read());
          }
        }
      #elif defined(OIDN_ARCH_XEHPC) || defined(OIDN_ARCH_XE2)
        #pragma unroll
        for (int boh = 0; boh < blockOH; ++boh)
        {
          accumRows[boh] = xmx::dpas<dpasDepth, dpasRepeat, float>(
            accumRows[boh],
            weightMat,
            inRows[(kh + boh) % blockOH].template select<blockOW * blockC, 1>(kw * blockC).read());
        }
      #else
        #pragma unroll
        for (int boh = 0; boh < blockOH; ++boh)
        {
          #pragma unroll
          for (int bow = 0; bow < blockOW; ++bow)
          {
            #pragma unroll
            for (int i = 0; i < blockC; ++i)
            {
              accumRows[boh].template select<blockC, 1>(bow * blockC) +=
                inRows[(kh + boh) % blockOH].template replicate_w<blockC, 1>((kw + bow) * blockC + i) *
                weightMat.template select<blockC, 1>(i * blockC);
            }
          }
        }
      #endif
      }
    }

    // Converts accumulator to output data type, applies bias and activation
    static oidn_inline void finalizeBlock(OutRows& outRows, AccumRows& accumRows, const SrcDstT* biasPtr)
    {
    #if defined(OIDN_ARCH_XEHPG)
      // Shuffle and down-convert accumulator rows to output rows
      #pragma unroll
      for (int boh = 0; boh < blockOH; ++boh)
      {
        auto outRowView = outRows[boh].template bit_cast_view<SrcDstT, blockOW, blockC>();
        #pragma unroll
        for (int bocb = 0; bocb < blockOCB; ++bocb)
          outRowView.template select<blockOW, 1, blockOC, 1>(0, bocb * blockOC) = accumRows[boh][bocb];
      }
    #else
      // Down-convert accumulator rows to output rows
      #pragma unroll
      for (int boh = 0; boh < blockOH; ++boh)
        outRows[boh] = accumRows[boh];
    #endif

      // Load bias vector
      const auto biasVec = loadBlock<SrcDstT, blockC>(biasPtr);

      // Add bias
      #pragma unroll
      for (int boh = 0; boh < blockOH; ++boh)
        outRows[boh] += biasVec.template replicate<blockOW>();

      // Apply activation
      //if (activation == Activation::ReLU)
      {
        #pragma unroll
        for (int boh = 0; boh < blockOH; ++boh)
          outRows[boh] = max(outRows[boh], simd<SrcDstT, blockOW * blockC>(0));
      }
    }

    // Returns the work-group size for the kernel, given the global size
    // For some architectures, the global size may need to be adjusted
    static WorkDim<3> getGroupSize(WorkDim<3>& globalSize, SYCLArch arch)
    {
      WorkDim<3> groupSize = {globalSize[0], 1, 1};

    #if defined(OIDN_ARCH_XEHPG)
      // Workaround for DPAS + EU fusion bug: make sure to have even number of threads per group
      if (globalSize[0] % 2 != 0 && globalSize[1] % 2 != 0 && globalSize[2] % 2 != 0)
      {
        // We can safely round up one of the spatial dimensions thanks to bounds checking in the kernel
        globalSize[2]++;
        groupSize[2]++;
      }
    #endif

      // Compute the final work-group size
    #if defined(OIDN_ARCH_XEHPC) || defined(OIDN_ARCH_XE2)
      const int maxGroupSize = (arch == SYCLArch::XeHPC) ? 32 : 8;
    #else
      const int maxGroupSize = 16;
    #endif

      for (; ;)
      {
        bool updated = false;

        // Try to increase one of the spatial dimensions (1 or 2), smallest first
        int dim = (groupSize[1] * blockOH < groupSize[2] * blockOW) ? 1 : 2;
        for (int i = 0; i < 2 && !updated; ++i, dim = 3-dim)
        {
          const int maxDiv = maxGroupSize / (groupSize[0] * groupSize[3-dim]);
          for (int div = groupSize[dim] + 1; div <= maxDiv && !updated; ++div)
          {
            if (globalSize[dim] % div == 0
              #if defined(OIDN_ARCH_XEHPG)
                && (groupSize[0] * groupSize[3-dim] * div) % 2 == 0 // must have even number of threads
              #endif
               )
            {
              groupSize[dim] = div;
              updated = true;
            }
          }
        }

        if (!updated)
          break;
      }

      return groupSize;
    }
  };

OIDN_ARCH_XE_NAMESPACE_END
OIDN_NAMESPACE_END