// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/output_process.h"
#include "core/tensor_accessor.h"
#include "core/image_accessor.h"
#include "core/color.h"
#include "core/tile.h"
#include "sycl_engine.h"
#include "sycl_common.h"

OIDN_NAMESPACE_BEGIN
OIDN_ARCH_XE_NAMESPACE_BEGIN

  template<typename SrcT, TensorLayout srcLayout>
  struct SYCLOutputProcessKernel
  {
    static constexpr int blockC = TensorByteOffset<SrcT, srcLayout>::blockC; // channel block size
    static constexpr int blockW = 16; // block width in pixels
    static constexpr int N = blockW;

    // Source
    TensorAccessor3D<SrcT, srcLayout> src;

    // Destination
    ImageAccessor dst;

    // Tile
    Tile tile;

    // Transfer function
    TransferFunction transferFunc;
    bool hdr;
    bool snorm; // signed normalized ([-1..1])

    oidn_inline void operator ()(const WorkGroupItem<2>& it) const SYCL_ESIMD_FUNCTION
    {
      const int h      = it.getGlobalID<0>();
      const int wBegin = it.getGlobalID<1>() * N;

      const int hSrc = h + tile.hSrcBegin;
      const int hDst = h + tile.hDstBegin;
      const int wSrcBegin = wBegin + tile.wSrcBegin;

      // Load
      simd<SrcT, N * blockC> inRow;
      loadRow(inRow, src, 0, hSrc, wSrcBegin);

      vec3<simd<float, N>> value(
        simd<float, N>(inRow.template select<N, blockC>(0)),
        simd<float, N>(inRow.template select<N, blockC>(1)),
        simd<float, N>(inRow.template select<N, blockC>(2)));

      // The CNN output may contain negative values or even NaNs, so it must be sanitized
      value = clamp(nan_to_zero(value), 0.f, FLT_MAX);

      // Apply the inverse transfer function
      value = transferFunc.inverse(value);

      // Average the channels if there is only one output channel
      if (dst.C == 1)
        value = (value.x + value.y + value.z) * (1.f / 3.f);

      // Sanitize
      if (snorm)
      {
        // Transform to [-1..1]
        value = value * 2.f - 1.f;
        value = max(value, -1.f);
      }
      if (!hdr)
        value = min(value, 1.f);

      // Scale
      value = value * transferFunc.getOutputScale();

      // Store
      const simd<int, N> w(wBegin, 1);
      const simd_mask<N> mask = w < tile.W;
      const simd<int, N> wDst = w + tile.wDstBegin;

      dst.set3<float, N>(hDst, wDst, value, mask);
    }
  };

  template<typename SrcT, TensorLayout srcLayout>
  class SYCLOutputProcess : public OutputProcess
  {
  public:
    SYCLOutputProcess(SYCLEngine* engine, const OutputProcessDesc& desc)
      : OutputProcess(desc),
        engine(engine) {}

    Engine* getEngine() const override { return engine; }

    void submitKernels(const Ref<CancellationToken>& ct) override
    {
      check();

      using Kernel = SYCLOutputProcessKernel<SrcT, srcLayout>;

      Kernel kernel;
      kernel.src = *src;
      kernel.dst = *dst;
      kernel.tile = tile;
      kernel.transferFunc = *transferFunc;
      kernel.hdr = hdr;
      kernel.snorm = snorm;

      const WorkDim<2> numGroups{tile.H, ceil_div(tile.W, Kernel::blockW)};
      const WorkDim<2> groupSize{1, 1};

      engine->submitESIMDKernel(numGroups, groupSize, kernel);
    }

  private:
    SYCLEngine* engine;
  };

  Ref<OutputProcess> newSYCLOutputProcess(SYCLEngine* engine, const OutputProcessDesc& desc)
  {
    return makeRef<SYCLOutputProcess<half, TensorLayout::Chw16c>>(engine, desc);
  }

OIDN_ARCH_XE_NAMESPACE_END
OIDN_NAMESPACE_END
