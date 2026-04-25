// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal_engine.h"
#include "core/concat_conv.h"

OIDN_NAMESPACE_BEGIN

  class MetalConcatConv2 final : public ConcatConv2
  {
  public:
    MetalConcatConv2(MetalEngine* engine, const ConcatConvDesc& desc);
    ~MetalConcatConv2();

    Engine* getEngine() const override { return engine; }

    void finalize() override;
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    void updateWeight() override;
    void updateBias() override;

    MetalEngine* engine;
    MPSGraph* mpsGraph = nullptr;
    MPSGraphTensor* mpsSrc0 = nullptr;
    MPSGraphTensor* mpsSrc1 = nullptr;
    MPSGraphTensor* mpsDst  = nullptr;
  };

OIDN_NAMESPACE_END
