// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/input_process.h"
#include "core/tensor_accessor.h"
#include "core/image_accessor.h"
#include "core/color.h"
#include "core/tile.h"
#include "sycl_engine.h"
#include "sycl_common.h"

OIDN_NAMESPACE_BEGIN
OIDN_ARCH_XE_NAMESPACE_BEGIN

  template<typename DstT, TensorLayout dstLayout>
  struct SYCLInputProcessKernel
  {
    static constexpr int blockC = TensorByteOffset<DstT, dstLayout>::blockC; // channel block size
    static constexpr int blockW = 16; // block width in pixels
    static constexpr int N = blockW;

    // Source
    ImageAccessor input;  // color, albedo or normal
    ImageAccessor albedo; // auxiliary albedo
    ImageAccessor normal; // auxiliary normal

    // Destination
    TensorAccessor3D<DstT, dstLayout> dst;

    // Tile
    Tile tile;

    // Transfer function
    TransferFunction transferFunc;
    bool hdr;
    bool snorm; // signed normalized ([-1..1])

    oidn_inline vec3<simd<float, N>> getInput(int h, simd<int, N> w, simd_mask<N> mask) const
    {
      vec3<simd<float, N>> value = input.get3<float, N>(h, w, mask);

      // Scale
      value = value * transferFunc.getInputScale();

      // Sanitize
      value = clamp(nan_to_zero(value), snorm ? -1.f : 0.f, hdr ? FLT_MAX : 1.f);

      if (snorm)
      {
        // Transform to [0..1]
        value = value * 0.5f + 0.5f;
      }

      // Apply the transfer function
      value = transferFunc.forward(value);

      value.merge(0.f, !mask);
      return value;
    }

    oidn_inline vec3<simd<float, N>> getAlbedo(int h, simd<int, N> w, simd_mask<N> mask) const
    {
      vec3<simd<float, N>> value = albedo.get3<float, N>(h, w, mask);

      // Sanitize
      value = clamp(nan_to_zero(value), 0.f, 1.f);

      value.merge(0.f, !mask);
      return value;
    }

    oidn_inline vec3<simd<float, N>> getNormal(int h, simd<int, N> w, simd_mask<N> mask) const
    {
      vec3<simd<float, N>> value = normal.get3<float, N>(h, w, mask);

      // Sanitize
      value = clamp(nan_to_zero(value), -1.f, 1.f);

      // Transform to [0..1]
      value = value * 0.5f + 0.5f;

      value.merge(0.f, !mask);
      return value;
    }

    oidn_inline void operator ()(const WorkGroupItem<2>& it) const SYCL_ESIMD_FUNCTION
    {
      const int hDst      = it.getGlobalID<0>();
      const int wDstBegin = it.getGlobalID<1>() * N;

      const int h = hDst - tile.hDstBegin;

      simd<DstT, N * blockC> outRow = 0;

      if (h >= 0 && h < tile.H)
      {
        const simd<int, N> wDst(wDstBegin, 1);
        const simd<int, N> w = wDst - tile.wDstBegin;
        const simd_mask<N> mask = (w >= 0) & (w < tile.W);

        const int          hSrc = h + tile.hSrcBegin;
        const simd<int, N> wSrc = w + tile.wSrcBegin;

        const vec3<simd<float, N>> inputValue = getInput(hSrc, wSrc, mask);
        outRow.template select<N, blockC>(0) = simd<DstT, N>(inputValue.x);
        outRow.template select<N, blockC>(1) = simd<DstT, N>(inputValue.y);
        outRow.template select<N, blockC>(2) = simd<DstT, N>(inputValue.z);

        if (albedo.ptr)
        {
          const vec3<simd<float, N>> albedoValue = getAlbedo(hSrc, wSrc, mask);
          outRow.template select<N, blockC>(3) = simd<DstT, N>(albedoValue.x);
          outRow.template select<N, blockC>(4) = simd<DstT, N>(albedoValue.y);
          outRow.template select<N, blockC>(5) = simd<DstT, N>(albedoValue.z);
        }

        if (normal.ptr)
        {
          const vec3<simd<float, N>> normalValue = getNormal(hSrc, wSrc, mask);
          outRow.template select<N, blockC>(6) = simd<DstT, N>(normalValue.x);
          outRow.template select<N, blockC>(7) = simd<DstT, N>(normalValue.y);
          outRow.template select<N, blockC>(8) = simd<DstT, N>(normalValue.z);
        }
      }

      storeRow(outRow, dst, 0, hDst, wDstBegin);
    }
  };

  template<typename DstT, TensorLayout dstLayout, int tensorBlockC>
  class SYCLInputProcess : public InputProcess
  {
  public:
    SYCLInputProcess(SYCLEngine* engine, const InputProcessDesc& desc)
      : InputProcess(engine, desc),
        engine(engine) {}

    Engine* getEngine() const override { return engine; }

    void submitKernels(const Ref<CancellationToken>& ct) override
    {
      check();

      using Kernel = SYCLInputProcessKernel<DstT, dstLayout>;

      Kernel kernel;
      Image nullImage;

      kernel.input  = color ? *color : (albedo ? *albedo : *normal);
      kernel.albedo = (color && albedo) ? *albedo : nullImage;
      kernel.normal = (color && normal) ? *normal : nullImage;
      kernel.dst    = *dst;
      kernel.tile   = tile;
      kernel.transferFunc = *transferFunc;
      kernel.hdr   = hdr;
      kernel.snorm = snorm;

      const WorkDim<2> numGroups{dst->getH(), ceil_div(dst->getW(), Kernel::blockW)};
      const WorkDim<2> groupSize{1, 1};

      engine->submitESIMDKernel(numGroups, groupSize, kernel);
    }

  private:
    SYCLEngine* engine;
  };

  Ref<InputProcess> newSYCLInputProcess(SYCLEngine* engine, const InputProcessDesc& desc)
  {
    return makeRef<SYCLInputProcess<half, TensorLayout::Chw16c, 16>>(engine, desc);
  }

OIDN_ARCH_XE_NAMESPACE_END
OIDN_NAMESPACE_END
