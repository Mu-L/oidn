// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "concat_conv_chw.h"
#include "engine.h"

OIDN_NAMESPACE_BEGIN

  ConcatConvCHW::ConcatConvCHW(Engine* engine, const ConcatConvDesc& desc)
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

  void ConcatConvCHW::updateSrc()
  {
    if (!src0->getBuffer() || !src1->getBuffer())
      throw std::invalid_argument("concat+conv sources must be backed by buffers");
    if (src0->getBuffer() != src1->getBuffer() ||
        (static_cast<char*>(src0->getPtr()) + src0->getByteSize()) != static_cast<char*>(src1->getPtr()))
      throw std::invalid_argument("concat+conv sources are not pre-concatenated in memory");

    auto src = src0->getBuffer()->newTensor(srcDesc, src0->getByteOffset());
    conv->setSrc(src);
  }

OIDN_NAMESPACE_END