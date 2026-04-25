// Copyright 2023 Intel Corporation
// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include "metal_conv.h"

OIDN_NAMESPACE_BEGIN

  MetalConv::MetalConv(MetalEngine* engine, const ConvDesc& desc)
    : Conv(desc),
      engine(engine)
  {}

  MetalConv::~MetalConv()
  {
    if (mpsGraph)
      [mpsGraph release];
  }

  void MetalConv::updateWeight()
  {
    if (mpsGraph)
      throw std::logic_error("convolution weight cannot be set after finalization");
    if (weight->getBuffer())
      throw std::invalid_argument("convolution weight must be a host tensor");
  }

  void MetalConv::updateBias()
  {
    if (mpsGraph)
      throw std::logic_error("convolution bias cannot be set after finalization");
    if (bias->getBuffer())
      throw std::invalid_argument("convolution bias must be a host tensor");
  }

  void MetalConv::finalize()
  {
    mpsGraph = [[MPSGraph alloc] init];

    mpsSrc = toMPSGraphPlaceholder(mpsGraph, srcDesc);

    MPSGraphTensor* mpsWeight = toMPSGraphConst(mpsGraph, weight);
    MPSGraphTensor* mpsBias   = toMPSGraphConst(mpsGraph, bias);

    MPSGraphTensor* mpsConvSrc = mpsSrc;
    if (fusion == Fusion::UpsampleSrc0)
    {
      mpsConvSrc = [mpsGraph resizeTensor: mpsSrc
                                     size: @[@(dstDesc.getH()), @(dstDesc.getW())]
                                     mode: MPSGraphResizeMode::MPSGraphResizeNearest
                             centerResult: true
                             alignCorners: false
                                   layout: MPSGraphTensorNamedDataLayout::MPSGraphTensorNamedDataLayoutNHWC
                                     name: nil];
    }

    MPSGraphConvolution2DOpDescriptor* mpsConvDesc = [MPSGraphConvolution2DOpDescriptor
      descriptorWithStrideInX: 1
                    strideInY: 1
              dilationRateInX: 1
              dilationRateInY: 1
                       groups: 1
                 paddingStyle: MPSGraphPaddingStyle::MPSGraphPaddingStyleTF_SAME
                   dataLayout: MPSGraphTensorNamedDataLayout::MPSGraphTensorNamedDataLayoutNHWC
                weightsLayout: MPSGraphTensorNamedDataLayout::MPSGraphTensorNamedDataLayoutOIHW];

    mpsDst = [mpsGraph convolution2DWithSourceTensor: mpsConvSrc
                                       weightsTensor: mpsWeight
                                          descriptor: mpsConvDesc
                                                name: nil];

    mpsDst = [mpsGraph additionWithPrimaryTensor: mpsDst
                                 secondaryTensor: mpsBias
                                            name: nil];

    if (activation == Activation::ReLU)
    {
      mpsDst = [mpsGraph reLUWithTensor: mpsDst
                                   name: nil];
    }

    if (fusion == Fusion::PoolDst)
    {
      MPSGraphPooling2DOpDescriptor* mpsPoolDesc = [MPSGraphPooling2DOpDescriptor
        descriptorWithKernelWidth: 2
                     kernelHeight: 2
                        strideInX: 2
                        strideInY: 2
                     paddingStyle: MPSGraphPaddingStyle::MPSGraphPaddingStyleTF_SAME
                       dataLayout: MPSGraphTensorNamedDataLayout::MPSGraphTensorNamedDataLayoutNHWC];

      mpsDst = [mpsGraph maxPooling2DWithSourceTensor: mpsDst
                                           descriptor: mpsPoolDesc
                                                 name: nil];
    }
  }

  void MetalConv::submitKernels(const Ref<CancellationToken>& ct)
  {
    MPSCommandBuffer* commandBuffer = engine->getMPSCommandBuffer();

    MPSGraphTensorData* mpsSrcData = newMPSGraphTensorData(src);
    MPSGraphTensorData* mpsDstData = newMPSGraphTensorData(dst);

    [mpsGraph encodeToCommandBuffer: commandBuffer
                              feeds: @{mpsSrc: mpsSrcData}
                   targetOperations: nil
                  resultsDictionary: @{mpsDst: mpsDstData}
                executionDescriptor: nil];

    [mpsSrcData release];
    [mpsDstData release];
  }

OIDN_NAMESPACE_END