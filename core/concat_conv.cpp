// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "concat_conv.h"

OIDN_NAMESPACE_BEGIN

  ConcatConvBase::ConcatConvBase(const ConcatConvDesc& desc)
    : ConcatConvDesc(desc)
  {
    if (src0Desc.getRank() != 3 ||
        src1Desc.getRank() != 3 ||
        src0Desc.layout != src1Desc.layout ||
        src0Desc.dataType != src1Desc.dataType)
      throw std::invalid_argument("invalid concat+conv source descriptor");

    switch (fusion)
    {
    case Fusion::None:
      if (src0Desc.getH() != src1Desc.getH() ||
          src0Desc.getW() != src1Desc.getW())
        throw std::invalid_argument("invalid concat+conv source descriptor");
      break;

    case Fusion::UpsampleSrc0: // src0 is upsampled by 2x
      if (src0Desc.getH() * 2 != src1Desc.getH() ||
          src0Desc.getW() * 2 != src1Desc.getW())
        throw std::invalid_argument("invalid concat+conv source descriptor");
      break;

    default:
      throw std::invalid_argument("unsupported concat+conv fusion");
    }

    if (weightDesc.getRank() != 4 || weightDesc.getI() != (src0Desc.getC() + src1Desc.getC()) ||
        weightDesc.getPaddedI() != (src0Desc.getPaddedC() + src1Desc.getPaddedC()))
      throw std::invalid_argument("invalid concat+conv weight shape");

    TensorDims dstDims{weightDesc.getO(), src1Desc.getH(), src1Desc.getW()};
    TensorDims dstPaddedDims{weightDesc.getPaddedO(), src1Desc.getH(), src1Desc.getW()};
    dstDesc = {dstDims, dstPaddedDims, src1Desc.layout, src1Desc.dataType};
  }

  void ConcatConvBase::setSrc(const Ref<Tensor>& src0, const Ref<Tensor>& src1)
  {
    if (!src0 || src0->getDesc() != src0Desc || !src1 || src1->getDesc() != src1Desc)
      throw std::invalid_argument("invalid concat+conv source");

    this->src0 = src0;
    this->src1 = src1;
    updateSrc();
  }

  void ConcatConvBase::setBias(const Ref<Tensor>& bias)
  {
    if (!bias || bias->getDesc() != biasDesc)
      throw std::invalid_argument("invalid concat+conv bias");

    this->bias = bias;
    updateBias();
  }

  void ConcatConvBase::setDst(const Ref<Tensor>& dst)
  {
    if (!dst || dst->getDesc() != dstDesc)
      throw std::invalid_argument("invalid concat+conv destination");

    this->dst = dst;
    updateDst();
  }

  // -----------------------------------------------------------------------------------------------

  void ConcatConv::setWeight(const Ref<Tensor>& weight)
  {
    if (!weight || weight->getDesc() != weightDesc)
      throw std::invalid_argument("invalid concat+conv weight");

    this->weight = weight;
    updateWeight();
  }

  // -----------------------------------------------------------------------------------------------

  ConcatConv2::ConcatConv2(const ConcatConvDesc& desc)
    : ConcatConvBase(desc)
  {
    if (src0Desc.layout != TensorLayout::hwc)
      throw std::logic_error("unsupported concat+conv source layout");
    if (fusion != Fusion::None && fusion != Fusion::UpsampleSrc0) // only pre-ops supported
      throw std::invalid_argument("unsupported concat+conv fusion");

    // Split the convolution into two smaller convolutions
    weight0Desc = {{dstDesc.getC(),       src0Desc.getC(),       weightDesc.getH(), weightDesc.getW()},
                   {dstDesc.getPaddedC(), src0Desc.getPaddedC(), weightDesc.getH(), weightDesc.getW()},
                   weightDesc.layout,
                   weightDesc.dataType};

    weight1Desc = {{dstDesc.getC(),       src1Desc.getC(),       weightDesc.getH(), weightDesc.getW()},
                   {dstDesc.getPaddedC(), src1Desc.getPaddedC(), weightDesc.getH(), weightDesc.getW()},
                   weightDesc.layout,
                   weightDesc.dataType};
  }

  void ConcatConv2::setWeight(const Ref<Tensor>& weight0, const Ref<Tensor>& weight1)
  {
    if (!weight0 || weight0->getDesc() != weight0Desc)
      throw std::invalid_argument("invalid concat+conv weight0");
    if (!weight1 || weight1->getDesc() != weight1Desc)
      throw std::invalid_argument("invalid concat+conv weight1");

    this->weight0 = weight0;
    this->weight1 = weight1;
    updateWeight();
  }

  // -----------------------------------------------------------------------------------------------

  PreConcatConvCHW::PreConcatConvCHW(Engine* engine, const ConcatConvDesc& desc)
    : ConcatConv(desc)
  {
    if (src0Desc.layout == TensorLayout::hwc)
      throw std::invalid_argument("unsupported concat+conv source layout");
    if (fusion != Fusion::None && fusion != Fusion::PoolDst) // only post-ops supported
      throw std::invalid_argument("unsupported concat+conv fusion");

    TensorDims srcDims{src0Desc.getC() + src1Desc.getC(), src0Desc.getH(), src0Desc.getW()};
    TensorDims srcPaddedDims{src0Desc.getPaddedC() + src1Desc.getPaddedC(), src0Desc.getH(), src0Desc.getW()};
    srcDesc = {srcDims, srcPaddedDims, src0Desc.layout, src0Desc.dataType};

    conv = engine->newConv({srcDesc, weightDesc, biasDesc, activation, fusion, fastMath});
  }

  void PreConcatConvCHW::updateSrc()
  {
    if (!src0->getBuffer() || !src1->getBuffer())
      throw std::invalid_argument("concat+conv sources must be backed by buffers");
    if (src0->getBuffer() != src1->getBuffer() ||
        (static_cast<char*>(src0->getPtr()) + src0->getByteSize()) != static_cast<char*>(src1->getPtr()))
      throw std::invalid_argument("concat+conv sources are not pre-concatenated in memory");

    auto src = src0->getBuffer()->newTensor(srcDesc, src0->getByteOffset());
    conv->setSrc(src);
  }

  // -----------------------------------------------------------------------------------------------

  InplaceConcatConv2::InplaceConcatConv2(Engine* engine, const ConcatConvDesc& desc)
    : ConcatConv2(desc)
  {
    // Convolution 0: dst = conv(src0, weight0) + bias
    conv0 = engine->newConv({src0Desc, weight0Desc, biasDesc, Activation::None, fusion, fastMath});

    // Convolution 1: dst = activation(conv(src1, weight1) + dst)
    // We use dst as bias
    conv1 = engine->newConv({src1Desc, weight1Desc, dstDesc, activation, Fusion::None, fastMath});
  }

  bool InplaceConcatConv2::isSupported() const
  {
    return conv0->isSupported() && conv1->isSupported();
  }

  size_t InplaceConcatConv2::getScratchByteSize()
  {
    return max(conv0->getScratchByteSize(), conv1->getScratchByteSize());
  }

  void InplaceConcatConv2::setScratch(const Ref<Buffer>& scratch)
  {
    conv0->setScratch(scratch);
    conv1->setScratch(scratch);
  }

  void InplaceConcatConv2::updateSrc()
  {
    conv0->setSrc(src0);
    conv1->setSrc(src1);
  }

  void InplaceConcatConv2::updateWeight()
  {
    conv0->setWeight(weight0);
    conv1->setWeight(weight1);
  }

  void InplaceConcatConv2::updateBias()
  {
    conv0->setBias(bias);
  }

  void InplaceConcatConv2::updateDst()
  {
    conv0->setDst(dst);

    conv1->setBias(dst);
    conv1->setDst(dst);
  }

  void InplaceConcatConv2::finalize()
  {
    conv0->finalize();
    conv1->finalize();
  }

  void InplaceConcatConv2::submitKernels(const Ref<CancellationToken>& ct)
  {
    conv0->submitKernels(ct);
    conv1->submitKernels(ct);
  }

OIDN_NAMESPACE_END
