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

    // Is pre-concatenation of the source tensors required?
    virtual bool isPreConcatConv() const { return false; }

    void setWeight(const Ref<Tensor>& weight);

  protected:
    virtual void updateWeight() {}

    Ref<Tensor> weight;
  };

  // Concatenation + convolution implemented as two separate convolutions
  class ConcatConv2 : public ConcatConvBase
  {
  public:
    ConcatConv2(const ConcatConvDesc& desc);

    TensorDesc getWeight0Desc() const { return weight0Desc; }
    TensorDesc getWeight1Desc() const { return weight1Desc; }
    void setWeight(const Ref<Tensor>& weight0, const Ref<Tensor>& weight1);

  protected:
    virtual void updateWeight() {}

    TensorDesc weight0Desc;
    TensorDesc weight1Desc;
    Ref<Tensor> weight0;
    Ref<Tensor> weight1;
  };

  // Concatenation + convolution for pre-concatenated CHW tensors (stored consecutively in memory)
  class PreConcatConvCHW final : public ConcatConv
  {
  public:
    PreConcatConvCHW(Engine* engine, const ConcatConvDesc& desc);

    Engine* getEngine() const override { return conv->getEngine(); }

    bool isPreConcatConv() const override { return true; }
    size_t getScratchByteSize() override { return conv->getScratchByteSize(); }
    void setScratch(const Ref<Buffer>& scratch) override { conv->setScratch(scratch); }

    void finalize() override { conv->finalize(); }
    void submitKernels(const Ref<CancellationToken>& ct) override { conv->submitKernels(ct); }

  private:
    void updateSrc() override;
    void updateWeight() override { conv->setWeight(weight); }
    void updateBias() override { conv->setBias(bias); }
    void updateDst() override { conv->setDst(dst); }

    TensorDesc srcDesc; // pre-concatenated source
    Ref<Conv> conv;
  };

  // Executes the two convolutions of ConcatConv2 in-place without an intermediate buffer for the
  // first convolution's output
  class InplaceConcatConv2 final : public ConcatConv2
  {
  public:
    InplaceConcatConv2(Engine* engine, const ConcatConvDesc& desc);

    Engine* getEngine() const override { return conv0->getEngine(); }
    bool isSupported() const override;

    size_t getScratchByteSize() override;
    void setScratch(const Ref<Buffer>& scratch) override;

    void finalize() override;
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    void updateSrc() override;
    void updateWeight() override;
    void updateBias() override;
    void updateDst() override;

    Ref<Conv> conv0;
    Ref<Conv> conv1;
  };

OIDN_NAMESPACE_END
