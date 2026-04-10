// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "conv.h"

OIDN_NAMESPACE_BEGIN

  // Concatenation + convolution descriptor
  struct ConcatConvDesc
  {
    TensorDesc src0Desc;
    TensorDesc src1Desc;
    TensorDesc weightDesc;
    TensorDesc biasDesc;
    Activation activation;
    Fusion fusion;
    bool fastMath; // prefer performance over accuracy
  };

  class ConcatConvBase : public BaseOp, protected ConcatConvDesc
  {
  public:
    ConcatConvBase(const ConcatConvDesc& desc);

    TensorDesc getDstDesc() const { return dstDesc; }
    Ref<Tensor> getDst() const { return dst; }

    void setSrc(const Ref<Tensor>& src0, const Ref<Tensor>& src1);
    void setBias(const Ref<Tensor>& bias);
    void setDst(const Ref<Tensor>& dst);

  protected:
    virtual void updateSrc() {}
    virtual void updateBias() {}
    virtual void updateDst() {}

    TensorDesc dstDesc;

    Ref<Tensor> src0;
    Ref<Tensor> src1;
    Ref<Tensor> bias;
    Ref<Tensor> dst;
  };

  class ConcatConv : public ConcatConvBase
  {
  public:
    ConcatConv(const ConcatConvDesc& desc) : ConcatConvBase(desc) {}

    void setWeight(const Ref<Tensor>& weight);

  protected:
    virtual void updateWeight() {}

  protected:
    Ref<Tensor> weight;
  };

OIDN_NAMESPACE_END
