// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "concat_conv_hwc.h"
#include "engine.h"

OIDN_NAMESPACE_BEGIN

  ConcatConvHWC::ConcatConvHWC(Engine* engine, const ConcatConvDesc& desc)
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

    // Convolution 0: dst = conv(src0, weight0) + bias
    conv0 = engine->newConv({src0Desc, weight0Desc, biasDesc, Activation::None, fusion, fastMath});

    // Convolution 1: dst = activation(conv(src1, weight1) + dst)
    // We use dst as bias
    conv1 = engine->newConv({src1Desc, weight1Desc, dstDesc, activation, Fusion::None, fastMath});
  }

  bool ConcatConvHWC::isSupported() const
  {
    return conv0->isSupported() && conv1->isSupported();
  }

  size_t ConcatConvHWC::getScratchByteSize()
  {
    return max(conv0->getScratchByteSize(), conv1->getScratchByteSize());
  }

  void ConcatConvHWC::setScratch(const Ref<Buffer>& scratch)
  {
    conv0->setScratch(scratch);
    conv1->setScratch(scratch);
  }

  void ConcatConvHWC::setWeight(const Ref<Tensor>& weight0, const Ref<Tensor>& weight1)
  {
    conv0->setWeight(weight0);
    conv1->setWeight(weight1);
  }

  void ConcatConvHWC::updateSrc()
  {
    conv0->setSrc(src0);
    conv1->setSrc(src1);
  }

  void ConcatConvHWC::updateDst()
  {
    conv0->setDst(dst);

    conv1->setBias(dst);
    conv1->setDst(dst);
  }

  void ConcatConvHWC::finalize()
  {
    conv0->finalize();
    conv1->finalize();
  }

  void ConcatConvHWC::submitKernels(const Ref<CancellationToken>& ct)
  {
    conv0->submitKernels(ct);
    conv1->submitKernels(ct);
  }

OIDN_NAMESPACE_END