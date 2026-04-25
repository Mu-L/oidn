// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "metal_concat_conv.h"
#include "metal_common.h"

OIDN_NAMESPACE_BEGIN

  MetalConcatConv2::MetalConcatConv2(MetalEngine* engine, const ConcatConvDesc& desc)
    : ConcatConv2(desc),
      engine(engine)
  {}

  MetalConcatConv2::~MetalConcatConv2()
  {
    if (mpsGraph)
      [mpsGraph release];
  }

  void MetalConcatConv2::updateWeight()
  {
    if (mpsGraph)
      throw std::logic_error("convolution weight cannot be set after finalization");
    if (weight0->getBuffer() || weight1->getBuffer())
      throw std::invalid_argument("convolution weight must be a host tensor");
  }

  void MetalConcatConv2::updateBias()
  {
    if (mpsGraph)
      throw std::logic_error("convolution bias cannot be set after finalization");
    if (bias->getBuffer())
      throw std::invalid_argument("convolution bias must be a host tensor");
  }

  void MetalConcatConv2::finalize()
  {
    mpsGraph = [[MPSGraph alloc] init];

    mpsSrc0 = toMPSGraphPlaceholder(mpsGraph, src0Desc);
    mpsSrc1 = toMPSGraphPlaceholder(mpsGraph, src1Desc);

    MPSGraphTensor* mpsWeight0 = toMPSGraphConst(mpsGraph, weight0);
    MPSGraphTensor* mpsWeight1 = toMPSGraphConst(mpsGraph, weight1);
    MPSGraphTensor* mpsBias    = toMPSGraphConst(mpsGraph, bias);

    MPSGraphConvolution2DOpDescriptor* mpsConvDesc = [MPSGraphConvolution2DOpDescriptor
      descriptorWithStrideInX: 1
                    strideInY: 1
              dilationRateInX: 1
              dilationRateInY: 1
                       groups: 1
                 paddingStyle: MPSGraphPaddingStyle::MPSGraphPaddingStyleTF_SAME
                   dataLayout: MPSGraphTensorNamedDataLayout::MPSGraphTensorNamedDataLayoutNHWC
                weightsLayout: MPSGraphTensorNamedDataLayout::MPSGraphTensorNamedDataLayoutOIHW];

    // Optionally upsample src0
    MPSGraphTensor* mpsConvSrc0 = mpsSrc0;
    if (fusion == Fusion::UpsampleSrc0)
    {
      mpsConvSrc0 = [mpsGraph resizeTensor: mpsSrc0
                                      size: @[@(dstDesc.getH()), @(dstDesc.getW())]
                                      mode: MPSGraphResizeMode::MPSGraphResizeNearest
                              centerResult: true
                              alignCorners: false
                                    layout: MPSGraphTensorNamedDataLayout::MPSGraphTensorNamedDataLayoutNHWC
                                      name: nil];
    }

    mpsDst = [mpsGraph convolution2DWithSourceTensor: mpsConvSrc0
                                       weightsTensor: mpsWeight0
                                          descriptor: mpsConvDesc
                                                name: nil];

    MPSGraphTensor* mpsTemp = [mpsGraph convolution2DWithSourceTensor: mpsSrc1
                                                        weightsTensor: mpsWeight1
                                                           descriptor: mpsConvDesc
                                                                 name: nil];

    mpsDst = [mpsGraph additionWithPrimaryTensor: mpsDst
                                 secondaryTensor: mpsTemp
                                            name: nil];

    mpsDst = [mpsGraph additionWithPrimaryTensor: mpsDst
                                 secondaryTensor: mpsBias
                                            name: nil];

    if (activation == Activation::ReLU)
    {
      mpsDst = [mpsGraph reLUWithTensor: mpsDst
                                   name: nil];
    }
  }

  void MetalConcatConv2::submitKernels(const Ref<CancellationToken>& ct)
  {
    MPSCommandBuffer* commandBuffer = engine->getMPSCommandBuffer();

    MPSGraphTensorData* mpsSrc0Data = newMPSGraphTensorData(src0);
    MPSGraphTensorData* mpsSrc1Data = newMPSGraphTensorData(src1);
    MPSGraphTensorData* mpsDstData  = newMPSGraphTensorData(dst);

    [mpsGraph encodeToCommandBuffer: commandBuffer
                              feeds: @{mpsSrc0: mpsSrc0Data, mpsSrc1: mpsSrc1Data}
                   targetOperations: nil
                  resultsDictionary: @{mpsDst: mpsDstData}
                executionDescriptor: nil];

    [mpsSrc0Data release];
    [mpsSrc1Data release];
    [mpsDstData release];
  }

OIDN_NAMESPACE_END
