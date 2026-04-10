// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concat_conv.h"

OIDN_NAMESPACE_BEGIN

  // Concatenation + convolution for HWC tensors
  // The convolution is split into two smaller convolutions, one for each input tensor
  // The weights for each convolution must be set separately
  class ConcatConvHWC final : public ConcatConvBase
  {
  public:
    ConcatConvHWC(Engine* engine, const ConcatConvDesc& desc);

    Engine* getEngine() const override { return conv0->getEngine(); }
    bool isSupported() const override;

    size_t getScratchByteSize() override;
    void setScratch(const Ref<Buffer>& scratch) override;

    TensorDesc getWeight0Desc() const { return weight0Desc; }
    TensorDesc getWeight1Desc() const { return weight1Desc; }
    void setWeight(const Ref<Tensor>& weight0, const Ref<Tensor>& weight1);

    void finalize() override;
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    void updateSrc() override;
    void updateBias() override { conv0->setBias(bias); }
    void updateDst() override;

    TensorDesc weight0Desc;
    TensorDesc weight1Desc;

    Ref<Conv> conv0;
    Ref<Conv> conv1;
  };

OIDN_NAMESPACE_END
