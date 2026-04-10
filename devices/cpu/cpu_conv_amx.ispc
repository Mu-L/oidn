// Copyright 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor_accessor.isph"
#include <amx.isph>

struct AMXTileConfig
{
  uniform uint8 palette_id;
  uniform uint8 start_row;
  uniform uint8 reserved_0[14];
  uniform uint16 colsb[16];
  uniform uint8 rows[16];
};

struct CPUConvAMXKernel
{
  uniform TensorAccessor3D_ChwBc src[2]; // one or two source tensors (optional concatenation)
  uniform TensorAccessor4D_OIhwPoQiRoSi weight;
  uniform TensorAccessor1D bias;
  uniform TensorAccessor3D_ChwBc dst;
  uniform int numSrcs; // number of source tensors to concatenate
  uniform int OH; // convolution output height without post-op
  uniform int OW; // convolution output width without post-op
  uniform bool relu;
  uniform Fusion fusion;

  uniform AMXTileConfig tileConfig; // set by CPUConvAMXKernel_init
};

#define KW 3 // kernel width
#define KH 3 // kernel height
#define PW 1 // padding width on each side
#define PH 1 // padding height on each side

#define blockOW 16                 // output block width (= AMX tile row count)
#define blockOH 2                  // output block height
#define blockIW (blockOW + KW - 1) // input block width (with borders for the kernel)
#define blockIH (blockOH + KH - 1) // input block height (with borders for the kernel)

#define wByteStride 64 // width byte stride of tensors (= AMX-512 vector size)

#if TARGET_WIDTH != blockOW
  #error "Unsupported ISPC target width"
#endif

// AMX tile mapping (7 tiles)
#define tmm_accumRow0_ocb0 ((uniform uint8)0) // accumulator row 0, output channel block 0
#define tmm_accumRow0_ocb1 ((uniform uint8)1) // accumulator row 0, output channel block 1
#define tmm_accumRow1_ocb0 ((uniform uint8)2) // accumulator row 1, output channel block 0
#define tmm_accumRow1_ocb1 ((uniform uint8)3) // accumulator row 1, output channel block 1
#define tmm_inRow0         ((uniform uint8)4) // input block row 0
#define tmm_inRow1         ((uniform uint8)5) // input block row 1
#define tmm_weight         ((uniform uint8)6) // weight output channel block 0/1

// Store output channel blocks
template<typename T>
inline unmasked void storeOCB(uniform int8* uniform ptr, varying T ocb0, varying T ocb1)
{
  // Use streaming stores to avoid polluting the caches
  streaming_store((uniform T* uniform)ptr,                       ocb0);
  streaming_store((uniform T* uniform)(ptr + sizeof(varying T)), ocb1);
}

// Prefetches a block of input data (including borders needed for the kernel) from a source
// tensor into L1 cache
inline unmasked void CPUConvAMXKernel_prefetchInputBlock(
                       const uniform TensorAccessor3D_ChwBc& src,
                       const uniform int8* uniform srcPtr)
{
  // Iterate over the input block height
  #pragma unroll
  for (uniform int bih = 0; bih < blockIH; ++bih)
  {
    // Iterate over the input block width
    #pragma unroll
    for (uniform int biw = 0; biw < blockIW; ++biw)
      prefetch_l1(srcPtr + biw * wByteStride);

    srcPtr += src.hByteStride; // next row in the source tensor
  }
}

// Copy a block of input data (including borders needed for the kernel) from a source tensor
// to a temporary buffer, handling zero-padding as needed
inline unmasked void CPUConvAMXKernel_copyInputBlock(
                       const uniform TensorAccessor3D_ChwBc& src,
                       uniform int ih, uniform int iw,
                       const uniform int8* uniform srcPtr,
                       varying int* uniform tempPtr)
{
  // Iterate over the input block height
  #pragma unroll
  for (uniform int bih = 0; bih < blockIH; ++bih)
  {
    // Iterate over the input block width
    #pragma unroll
    for (uniform int biw = 0; biw < blockIW; ++biw)
    {
      // Bounds check
      if (ih + bih >= 0 && ih + bih < src.H &&
          iw + biw >= 0 && iw + biw < src.W)
        *tempPtr = *((const varying int* uniform)(srcPtr + biw * wByteStride));
      else
        *tempPtr = 0; // zero-padding

      ++tempPtr;
    }

    srcPtr += src.hByteStride; // next row in the source tensor
  }
}

// Copy a block of input data (including borders needed for the kernel) from a virtually
// 2x upsampled source tensor to a temporary buffer, handling zero-padding as needed
// Assumes: ih%2 == PH%2 and iw%2 == PW%2
inline unmasked void CPUConvAMXKernel_upsampleInputBlock(
                       const uniform TensorAccessor3D_ChwBc& src,
                       uniform int ic, uniform int ih, uniform int iw,
                       const uniform int8* uniform srcPtr,
                       varying int* uniform tempPtr)
{
  // Iterate over the input block height
  #pragma unroll
  for (uniform int bih = 0; bih < blockIH; ++bih)
  {
    // Iterate over the input block width
    #pragma unroll
    for (uniform int biw = 0; biw < blockIW; ++biw)
    {
      // Bounds check
      if (ih + bih >= 0 && ih + bih < src.H * 2 &&
          iw + biw >= 0 && iw + biw < src.W * 2)
        *tempPtr = *((const varying int* uniform)(srcPtr + ((biw + (PW & 1)) >> 1) * wByteStride));
      else
        *tempPtr = 0; // zero-padding

      ++tempPtr;
    }

    if ((bih + PH) & 1)
      srcPtr += src.hByteStride; // next row in the source tensor
  }
}

// Copy a block of input data (including borders needed for the kernel) from a virtually
// 2x upsampled source tensor to a temporary buffer, assuming that no padding is needed
// Assumes: ih%2 == PH%2 and iw%2 == PW%2
inline unmasked void CPUConvAMXKernel_upsampleInputInnerBlock(
                       const uniform TensorAccessor3D_ChwBc& src,
                       uniform int ic, uniform int ih, uniform int iw,
                       const uniform int8* uniform srcPtr,
                       varying int* uniform tempPtr)
{
  // Iterate over the input block height
  #pragma unroll
  for (uniform int bih = 0; bih < blockIH; ++bih)
  {
    // Iterate over the input block width
    #pragma unroll
    for (uniform int biw = 0; biw < blockIW; ++biw)
      *tempPtr++ = *((const varying int* uniform)(srcPtr + ((biw + (PW & 1)) >> 1) * wByteStride));

    if ((bih + PH) & 1)
       srcPtr += src.hByteStride;
  }
}

inline unmasked void CPUConvAMXKernel_computeBlock_f16(
                       const uniform CPUConvAMXKernel* uniform self,
                       uniform int oc, uniform int oh, uniform int ow)
{
  const uniform int blockIC = 32; // input channel block size

  // Temporary buffer for the input block (including borders needed for the kernel)
  // Input blocks which require padding must be copied here first from the source tensor
  varying int inBuffer[blockIH * blockIW]; // int is opaque type here

  // AMX accumulator tiles must be stored to temporary memory before loading them into AVX registers
  // Each tile holds a 16 element wide row and only a block of 16 output channels, so multiple tiles
  // are needed to hold all channels of the result
  varying float accumRow0_ocb0[blockOW]; // accumulator row 0, output channels 0..15
  varying float accumRow0_ocb1[blockOW]; // accumulator row 0, output channels 16..31
  varying float accumRow1_ocb0[blockOW]; // accumulator row 1, output channels 0..15
  varying float accumRow1_ocb1[blockOW]; // accumulator row 1, output channels 16..31

  // Initialize the accumulator tiles to zero
  amx_tile_zero(tmm_accumRow0_ocb0);
  amx_tile_zero(tmm_accumRow0_ocb1);
  amx_tile_zero(tmm_accumRow1_ocb0);
  amx_tile_zero(tmm_accumRow1_ocb1);

  const uniform int8* uniform weightPtr = Tensor_getPtr<float16>(self->weight, oc);

  // Iterate over the source tensors
  for (uniform int i = 0; i < self->numSrcs; ++i)
  {
    const uniform TensorAccessor3D_ChwBc& src = self->src[i];

    // Iterate over the input channel blocks
    for (uniform int ic = 0; ic < src.C; ic += blockIC)
    {
      uniform int8* uniform inPtr;
      uniform size_t inRowByteStride;

      const uniform int ih = oh - PH;
      const uniform int iw = ow - PW;

      if (i == 0 && self->fusion == Fusion_UpsampleSrc0)
      {
        // Upsample by 2x2 nearest-neighbor
        // Get a pointer to the input block in the source tensor
        // Tensor_get doesn't work with negative indices, so we need to adjust the pointer
        const uniform int8* uniform srcPtr = Tensor_getPtr<float16>(src, ic, oh >> 1, ow >> 1);
        srcPtr -= ((PH + 1) >> 1) * src.hByteStride; // adjust for top padding
        srcPtr -= ((PW + 1) >> 1) * wByteStride;     // adjust for left padding

        const uniform bool isInnerBlock = (ih >= 0) && (ih + blockIH <= src.H * 2) &&
                                          (iw >= 0) && (iw + blockIW <= src.W * 2);
        if (isInnerBlock)
          CPUConvAMXKernel_upsampleInputInnerBlock(src, ic, ih, iw, srcPtr, inBuffer);
        else
          CPUConvAMXKernel_upsampleInputBlock(src, ic, ih, iw, srcPtr, inBuffer);
        inPtr = (uniform int8* uniform)&inBuffer;
        inRowByteStride = blockIW * wByteStride;
      }
      else
      {
        // Get a pointer to the input block in the source tensor
        // Tensor_get doesn't work with negative indices, so we need to adjust the pointer
        const uniform int8* uniform srcPtr = Tensor_getPtr<float16>(src, ic, oh, ow);
        srcPtr -= PH * src.hByteStride; // adjust for top padding
        srcPtr -= PW * wByteStride;     // adjust for left padding

        // Prefetch or copy+pad the input block, depending on whether it is an inner block
        const uniform bool isInnerBlock = (ih >= 0) && (ih + blockIH <= src.H) &&
                                          (iw >= 0) && (iw + blockIW <= src.W);
        if (isInnerBlock)
        {
          // No padding needed, just prefetch from the source tensor (fast)
          CPUConvAMXKernel_prefetchInputBlock(src, srcPtr);
          inPtr = srcPtr;
          inRowByteStride = src.hByteStride;
        }
        else
        {
          // Padding needed, copy to the temporary buffer (slow)
          CPUConvAMXKernel_copyInputBlock(src, ih, iw, srcPtr, inBuffer);
          inPtr = (uniform int8* uniform)&inBuffer;
          inRowByteStride = blockIW * wByteStride;
        }
      }

      // Iterate over the kernel height
      #pragma unroll
      for (uniform int kh = 0; kh < KH; ++kh)
      {
        const uniform int8* uniform inRowPtr = inPtr; // current input block row

        // Iterate over the kernel width
        #pragma unroll
        for (uniform int kw = 0; kw < KW; ++kw)
        {
          const uniform bool isLast = (kh == KH - 1) && (kw == KW - 1);

          // Load weights for output channel block 0
          amx_tile_load(tmm_weight, weightPtr, 64);

          // Load input block row 0
          amx_tile_load(tmm_inRow0, inRowPtr, 64);
          // Multiply-accumulate block row 0, output channel block 0
          amx_dpfp16ps(tmm_accumRow0_ocb0, tmm_inRow0, tmm_weight);
          if (isLast)
            amx_tile_store(tmm_accumRow0_ocb0, (uniform int8* uniform)&accumRow0_ocb0, 64);

          // Load input block row 1
          amx_tile_load(tmm_inRow1, inRowPtr + inRowByteStride, 64);
          // Multiply-accumulate block row 1, output channel block 0
          amx_dpfp16ps(tmm_accumRow1_ocb0, tmm_inRow1, tmm_weight);
          if (isLast)
            amx_tile_store(tmm_accumRow1_ocb0, (uniform int8* uniform)&accumRow1_ocb0, 64);

          // Load weights for output channel block 1
          amx_tile_load(tmm_weight, weightPtr + 1024, 64);

          // Multiply-accumulate block row 0, output channel block 1
          amx_dpfp16ps(tmm_accumRow0_ocb1, tmm_inRow0, tmm_weight);
          if (isLast)
            amx_tile_store(tmm_accumRow0_ocb1, (uniform int8* uniform)&accumRow0_ocb1, 64);

          // Multiply-accumulate block row 1, output channel block 1
          amx_dpfp16ps(tmm_accumRow1_ocb1, tmm_inRow1, tmm_weight);
          if (isLast)
            amx_tile_store(tmm_accumRow1_ocb1, (uniform int8* uniform)&accumRow1_ocb1, 64);

          inRowPtr  += wByteStride; // shift input block by one column
          weightPtr += 2048;        // next pair of weight matrix tiles
        }

        inPtr += inRowByteStride; // shift input block by one row
      }
    }
  }

  // Load the bias
  const varying float16* uniform biasPtr =
    (varying float16* uniform)Tensor_getPtr<float16>(self->bias, oc);
  const varying float bias_ocb0 = biasPtr[0];
  const varying float bias_ocb1 = biasPtr[1];

  // Compute and store the final results to the destination tensor
  const uniform int validBlockOW = min(blockOW, self->OW - ow); // check width bounds

  if (self->fusion == Fusion_PoolDst)
  {
    // Downsample by 2x2 max pooling
    uniform int8* uniform dstRow = Tensor_getPtr<float16>(self->dst, oc, oh >> 1, ow >> 1);

    for (uniform int bow = 0; bow < validBlockOW; bow += 2)
    {
      // Load the accumulated values
      varying float x00_ocb0 = accumRow0_ocb0[bow];
      varying float x00_ocb1 = accumRow0_ocb1[bow];
      varying float x10_ocb0 = accumRow1_ocb0[bow];
      varying float x10_ocb1 = accumRow1_ocb1[bow];
      varying float x01_ocb0 = accumRow0_ocb0[bow+1];
      varying float x01_ocb1 = accumRow0_ocb1[bow+1];
      varying float x11_ocb0 = accumRow1_ocb0[bow+1];
      varying float x11_ocb1 = accumRow1_ocb1[bow+1];

      // Add the bias
      x00_ocb0 += bias_ocb0;  x00_ocb1 += bias_ocb1;
      x10_ocb0 += bias_ocb0;  x10_ocb1 += bias_ocb1;
      x01_ocb0 += bias_ocb0;  x01_ocb1 += bias_ocb1;
      x11_ocb0 += bias_ocb0;  x11_ocb1 += bias_ocb1;

      // Apply ReLU activation
      if (self->relu)
      {
        x00_ocb0 = max(x00_ocb0, 0);  x00_ocb1 = max(x00_ocb1, 0);
        x10_ocb0 = max(x10_ocb0, 0);  x10_ocb1 = max(x10_ocb1, 0);
        x01_ocb0 = max(x01_ocb0, 0);  x01_ocb1 = max(x01_ocb1, 0);
        x11_ocb0 = max(x11_ocb0, 0);  x11_ocb1 = max(x11_ocb1, 0);
      }

      // Max pooling over 2x2 region
      varying float x_ocb0 = max(max(x00_ocb0, x01_ocb0), max(x10_ocb0, x11_ocb0));
      varying float x_ocb1 = max(max(x00_ocb1, x01_ocb1), max(x10_ocb1, x11_ocb1));

      // Convert the result and store to the destination tensor
      storeOCB(dstRow, (float16)x_ocb0, (float16)x_ocb1);

      // Next destination column
      dstRow += wByteStride;
    }
  }
  else
  {
    uniform int8* uniform dstRow0 = Tensor_getPtr<float16>(self->dst, oc, oh, ow);
    uniform int8* uniform dstRow1 = dstRow0 + self->dst.hByteStride;

    for (uniform int bow = 0; bow < validBlockOW; ++bow)
    {
      // Load the accumulated values
      varying float x0_ocb0 = accumRow0_ocb0[bow];
      varying float x0_ocb1 = accumRow0_ocb1[bow];
      varying float x1_ocb0 = accumRow1_ocb0[bow];
      varying float x1_ocb1 = accumRow1_ocb1[bow];

      // Add the bias
      x0_ocb0 += bias_ocb0;  x0_ocb1 += bias_ocb1;
      x1_ocb0 += bias_ocb0;  x1_ocb1 += bias_ocb1;

      // Apply ReLU activation
      if (self->relu)
      {
        x0_ocb0 = max(x0_ocb0, 0);  x0_ocb1 = max(x0_ocb1, 0);
        x1_ocb0 = max(x1_ocb0, 0);  x1_ocb1 = max(x1_ocb1, 0);
      }

      // Convert the result and store to the destination tensor
      storeOCB(dstRow0, (float16)x0_ocb0, (float16)x0_ocb1);
      if (oh + 1 < self->OH) // check height bounds
        storeOCB(dstRow1, (float16)x1_ocb0, (float16)x1_ocb1);

      // Next destination column
      dstRow0 += wByteStride;
      dstRow1 += wByteStride;
    }
  }
}

// Initialize the AMX tile configuration data structure
export void CPUConvAMXKernel_init(uniform CPUConvAMXKernel* uniform self)
{
  memset((uniform int8* uniform)&self->tileConfig, 0, sizeof(AMXTileConfig));

  self->tileConfig.palette_id = 1;
  self->tileConfig.start_row = 0;

  // Configure 7 tiles
  for (uniform int i = 0; i < 7; ++i)
  {
    self->tileConfig.colsb[i] = 64; // bytes per row
    self->tileConfig.rows[i]  = 16;
  }
}

export void CPUConvAMXKernel_run_f16(
              const uniform CPUConvAMXKernel* uniform self,
              uniform int oc,
              uniform int ohBegin, uniform int ohEnd,
              uniform int ow,
              uniform bool isFirst, uniform bool isLast)
{
  if (isFirst)
    amx_tile_loadconfig((const uniform int8* uniform)&self->tileConfig);

  for (uniform int oh = ohBegin; oh < ohEnd; oh += blockOH)
    CPUConvAMXKernel_computeBlock_f16(self, oc, oh, ow);

  if (isLast)
    amx_tile_release();
}