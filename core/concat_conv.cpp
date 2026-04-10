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

  void ConcatConv::setWeight(const Ref<Tensor>& weight)
  {
    if (!weight || weight->getDesc() != weightDesc)
      throw std::invalid_argument("invalid concat+conv weight");

    this->weight = weight;
    updateWeight();
  }

OIDN_NAMESPACE_END
