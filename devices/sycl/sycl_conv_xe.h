// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sycl_conv_base_xe.h"

OIDN_NAMESPACE_BEGIN
OIDN_ARCH_XE_NAMESPACE_BEGIN

  template<typename SrcDstT, typename WeightT, TensorLayout srcDstLayout, TensorLayout weightLayout,
           Fusion fusion>
  struct SYCLConvKernel : SYCLConvBaseKernel<SrcDstT, WeightT, srcDstLayout, weightLayout>
  {
    using Base = SYCLConvBaseKernel<SrcDstT, WeightT, srcDstLayout, weightLayout>;
    using Base::KH, Base::KW, Base::PH, Base::PW;
    using Base::blockC, Base::blockOH, Base::blockOW;
    using typename Base::AccumRows, typename Base::InRows, typename Base::OutRows;

    TensorAccessor3D<SrcDstT, srcDstLayout> src;
    TensorAccessor4D<WeightT, weightLayout> weight;
    TensorAccessor1D<SrcDstT> bias;
    TensorAccessor3D<SrcDstT, srcDstLayout> dst;
    //Activation activation;

    oidn_inline void operator ()(const WorkGroupItem<3>& it) const SYCL_ESIMD_FUNCTION
    {
      AccumRows accumRows = {}; // = 0
      InRows inRows; // ring buffer for input rows

      const int oc = it.getLocalID<0>()  * blockC;
      const int oh = it.getGlobalID<1>() * blockOH;
      const int ow = it.getGlobalID<2>() * blockOW;

      // Iterate over input channel blocks
      for (int ic = 0; ic < src.C; ic += blockC)
      {
        const int ih = oh - PH;
        const int iw = ow - PW;

        // Load input rows into a ring buffer
        Base::template loadRows<blockOH - 1>(inRows, src, ic, ih, iw);

        // Iterate over kernel height
        #pragma unroll
        for (int kh = 0; kh < KH; ++kh)
        {
          // Load next input row into ring buffer
          Base::loadRow(inRows[(kh + blockOH - 1) % blockOH], src, ic, ih + (kh + blockOH - 1), iw);

          // Multiply + accumulate kernel row
          Base::mmaBlockKW(accumRows, inRows, &weight(oc, ic, kh, 0), kh);
        }
      }

      // Convert accumulator to output data type, apply bias and activation
      OutRows outRows;
      Base::finalizeBlock(outRows, accumRows, &bias(oc));

      // Store output rows
      if constexpr (fusion == Fusion::PoolDst)
      {
        #pragma unroll
        for (int boh = 0; boh < blockOH; boh += 2)
        {
          if (oh + boh >= src.H) // src.H = output height before pooling
            break;

          // Pool output rows
          auto poolRow2x1 = max(outRows[boh], outRows[boh + 1]);
          auto poolRow2x2 = max(poolRow2x1.template replicate_vs_w<blockOW / 2, blockC * 2, blockC>(0),
                                poolRow2x1.template replicate_vs_w<blockOW / 2, blockC * 2, blockC>(blockC));

          // Store pooled row
          Base::storeRow(poolRow2x2, dst, oc, (oh + boh) / 2, ow / 2);
        }
      }
      else
      {
        // Store output rows
        Base::template storeRows<blockOH>(outRows, dst, oc, oh, ow);
      }
    }
  };

  template<typename SrcDstT, Fusion kernelFusion>
  class SYCLConv : public Conv
  {
  public:
    SYCLConv(SYCLEngine* engine, const ConvDesc& desc)
      : Conv(desc),
        engine(engine)
    {
      if (srcDesc.layout != srcDstLayout || srcDesc.dataType != DataTypeOf<SrcDstT>::value)
        throw std::invalid_argument("unsupported convolution source layout/data type");
      if (weightDesc.getW() != 3 || weightDesc.getH() != 3)
        throw std::invalid_argument("unsupported convolution kernel size");
      if (weightDesc.layout != weightLayout || weightDesc.dataType != DataTypeOf<WeightT>::value)
        throw std::invalid_argument("unsupported convolution weight layout/data type");
      if (biasDesc.layout != TensorLayout::x || biasDesc.dataType != DataTypeOf<SrcDstT>::value)
        throw std::invalid_argument("unsupported convolution bias layout/data type");

      if (desc.activation != Activation::ReLU)
        throw std::invalid_argument("unsupported convolution activation");
      if (desc.fusion != kernelFusion)
        throw std::invalid_argument("unsupported convolution fusion");
    }

    Engine* getEngine() const override { return engine; }

    void submitKernels(const Ref<CancellationToken>& ct) override
    {
      if (!src || !weight || !bias || !dst)
        throw std::logic_error("convolution argument not set");

      using Kernel = SYCLConvKernel<SrcDstT, WeightT, srcDstLayout, weightLayout, kernelFusion>;

      Kernel kernel;
      kernel.src    = *src;
      kernel.weight = *weight;
      kernel.bias   = *bias;
      kernel.dst    = *dst;
      //kernel.activation = activation;

      WorkDim<3> globalSize = {dst->getPaddedC() / Kernel::blockC,
                               ceil_div(src->getH(), Kernel::blockOH),
                               ceil_div(src->getW(), Kernel::blockOW)};

      WorkDim<3> groupSize = Kernel::getGroupSize(globalSize, engine->getArch());

    #if defined(OIDN_ARCH_XEHPG)
      engine->submitESIMDKernelWithLargeGRF(globalSize / groupSize, groupSize, kernel);
    #else
      engine->submitESIMDKernel(globalSize / groupSize, groupSize, kernel);
    #endif
    }

  private:
    static constexpr TensorLayout srcDstLayout = TensorLayout::Chw16c;

  #if defined(OIDN_ARCH_XEHPG)
    using WeightT = SrcDstT;
    static constexpr TensorLayout weightLayout = TensorLayout::OIhw2o8i8o2i;
  #elif defined(OIDN_ARCH_XEHPC) || defined(OIDN_ARCH_XE2)
    using WeightT = SrcDstT;
    static constexpr TensorLayout weightLayout = TensorLayout::OIhw8i16o2i;
  #else
    using WeightT = float;
    static constexpr TensorLayout weightLayout = TensorLayout::OIhw16i16o;
  #endif

    SYCLEngine* engine;
  };

  Ref<Conv> newSYCLConv(SYCLEngine* engine, const ConvDesc& desc)
  {
    switch (desc.fusion)
    {
    case Fusion::None:
      return makeRef<SYCLConv<half, Fusion::None>>(engine, desc);
    case Fusion::PoolDst:
      return makeRef<SYCLConv<half, Fusion::PoolDst>>(engine, desc);
    default:
      throw std::invalid_argument("unsupported convolution fusion");
    }
  }

OIDN_ARCH_XE_NAMESPACE_END
OIDN_NAMESPACE_END