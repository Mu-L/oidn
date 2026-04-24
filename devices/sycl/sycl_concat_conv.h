// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sycl_conv_base.h"

OIDN_NAMESPACE_BEGIN
OIDN_ARCH_XE_NAMESPACE_BEGIN

  template<typename SrcDstT, typename WeightT, TensorLayout srcDstLayout, TensorLayout weightLayout>
  struct SYCLUpsampleConcatConvKernel : SYCLConvBaseKernel<SrcDstT, WeightT, srcDstLayout, weightLayout>
  {
    using Base = SYCLConvBaseKernel<SrcDstT, WeightT, srcDstLayout, weightLayout>;
    using Base::KH, Base::KW, Base::PH, Base::PW;
    using Base::blockC, Base::blockOH, Base::blockOW;
    using typename Base::AccumRows, typename Base::InRows, typename Base::OutRows;

    TensorAccessor3D<SrcDstT, srcDstLayout> src0, src1;
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

      const int ih = oh - PH;
      const int iw = ow - PW;

      // Iterate over src0 input channel blocks, while 2x upsampling src0
      for (int ic = 0; ic < src0.C; ic += blockC)
      {
        // Load input rows into ring buffer, upsampling by 2x
        #pragma unroll
        for (int boh = 0; boh < blockOH - 1; ++boh)
        {
          if (boh == 0 || (boh + PH) % 2 == 0) // only need to load every other row due to 2x upsampling
            loadUpsampleRow<PW & 1>(inRows[boh], src0, ic, ih + boh, iw);
          else
            inRows[boh] = inRows[boh - 1]; // duplicate previous row
        }

        // Iterate over kernel height
        #pragma unroll
        for (int kh = 0; kh < KH; ++kh)
        {
          // Load next input row into ring buffer, upsampling by 2x
          if ((kh + blockOH - 1 + PH) % 2 == 0) // only need to load every other row due to 2x upsampling
            loadUpsampleRow<PW & 1>(inRows[(kh + blockOH - 1) % blockOH], src0, ic, ih + kh + blockOH - 1, iw);
          else
            inRows[(kh + blockOH - 1) % blockOH] = inRows[(kh + blockOH - 2) % blockOH]; // duplicate previous row

          // Multiply + accumulate kernel row
          Base::mmaBlockKW(accumRows, inRows, &weight(oc, ic, kh, 0), kh);
        }
      }

      // Iterate over src1 input channel blocks
      for (int ic = 0; ic < src1.C; ic += blockC)
      {
        // Load input rows into ring buffer
        loadRows<blockOH - 1>(inRows, src1, ic, ih, iw);

        // Iterate over kernel height
        #pragma unroll
        for (int kh = 0; kh < KH; ++kh)
        {
          // Load next input row into ring buffer
          loadRow(inRows[(kh + blockOH - 1) % blockOH], src1, ic, ih + kh + blockOH - 1, iw);

          // Multiply + accumulate kernel row
          Base::mmaBlockKW(accumRows, inRows, &weight(oc, src0.C + ic, kh, 0), kh);
        }
      }

      // Convert accumulator to output data type, apply bias and activation
      OutRows outRows;
      Base::finalizeBlock(outRows, accumRows, &bias(oc));

      // Store output rows
      storeRows<blockOH>(outRows, dst, oc, oh, ow);
    }
  };

  template<typename SrcDstT>
  class SYCLUpsampleConcatConv : public ConcatConv
  {
  public:
    SYCLUpsampleConcatConv(SYCLEngine* engine, const ConcatConvDesc& desc)
      : ConcatConv(desc),
        engine(engine)
    {
      if (src0Desc.layout != srcDstLayout || src0Desc.dataType != DataTypeOf<SrcDstT>::value)
        throw std::invalid_argument("unsupported convolution source layout/data type");
      if (src1Desc.layout != srcDstLayout || src1Desc.dataType != DataTypeOf<SrcDstT>::value)
        throw std::invalid_argument("unsupported convolution source layout/data type");
      if (weightDesc.getW() != 3 || weightDesc.getH() != 3)
        throw std::invalid_argument("unsupported convolution kernel size");
      if (weightDesc.layout != weightLayout || weightDesc.dataType != DataTypeOf<WeightT>::value)
        throw std::invalid_argument("unsupported convolution weight layout/data type");
      if (biasDesc.layout != TensorLayout::x || biasDesc.dataType != DataTypeOf<SrcDstT>::value)
        throw std::invalid_argument("unsupported convolution bias layout/data type");

      if (desc.activation != Activation::ReLU)
        throw std::invalid_argument("unsupported convolution activation");
      if (desc.fusion != Fusion::UpsampleSrc0)
        throw std::invalid_argument("unsupported convolution fusion");
    }

    Engine* getEngine() const override { return engine; }

    void submitKernels(const Ref<CancellationToken>& ct) override
    {
      if (!src0 || !src1 || !weight || !bias || !dst)
        throw std::logic_error("concat+convolution argument not set");

      submitImpl();
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

    void submitImpl()
    {
      using Kernel = SYCLUpsampleConcatConvKernel<SrcDstT, WeightT, srcDstLayout, weightLayout>;

      Kernel kernel;
      kernel.src0   = *src0;
      kernel.src1   = *src1;
      kernel.weight = *weight;
      kernel.bias   = *bias;
      kernel.dst    = *dst;
      //kernel.activation = activation;

      WorkDim<3> globalSize = {dst->getPaddedC() / Kernel::blockC,
                               ceil_div(dst->getH(), Kernel::blockOH),
                               ceil_div(dst->getW(), Kernel::blockOW)};

      WorkDim<3> groupSize = Kernel::getGroupSize(globalSize);

    #if defined(OIDN_ARCH_XEHPG)
      engine->submitESIMDKernelWithLargeGRF(globalSize / groupSize, groupSize, kernel);
    #else
      engine->submitESIMDKernel(globalSize / groupSize, groupSize, kernel);
    #endif
    }

    SYCLEngine* engine;
  };

  Ref<ConcatConv> newSYCLConcatConv(SYCLEngine* engine, const ConcatConvDesc& desc)
  {
    return makeRef<SYCLUpsampleConcatConv<half>>(engine, desc);
  }

OIDN_ARCH_XE_NAMESPACE_END
OIDN_NAMESPACE_END