// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/input_process.h"
#include "core/output_process.h"
#include "core/conv.h"
#include "core/concat_conv.h"
#include "sycl_engine.h"

OIDN_NAMESPACE_BEGIN

  namespace xelp
  {
    Ref<InputProcess> newSYCLInputProcess(SYCLEngine* engine, const InputProcessDesc& desc);
    Ref<OutputProcess> newSYCLOutputProcess(SYCLEngine* engine, const OutputProcessDesc& desc);
    Ref<Conv> newSYCLConv(SYCLEngine* engine, const ConvDesc& desc);
    Ref<ConcatConv> newSYCLConcatConv(SYCLEngine* engine, const ConcatConvDesc& desc);
  }

  namespace xehpg
  {
    Ref<InputProcess> newSYCLInputProcess(SYCLEngine* engine, const InputProcessDesc& desc);
    Ref<OutputProcess> newSYCLOutputProcess(SYCLEngine* engine, const OutputProcessDesc& desc);
    Ref<Conv> newSYCLConv(SYCLEngine* engine, const ConvDesc& desc);
    Ref<ConcatConv> newSYCLConcatConv(SYCLEngine* engine, const ConcatConvDesc& desc);
  }

#if defined(__linux__)
  namespace xehpc
  {
    Ref<InputProcess> newSYCLInputProcess(SYCLEngine* engine, const InputProcessDesc& desc);
    Ref<OutputProcess> newSYCLOutputProcess(SYCLEngine* engine, const OutputProcessDesc& desc);
    Ref<Conv> newSYCLConv(SYCLEngine* engine, const ConvDesc& desc);
    Ref<ConcatConv> newSYCLConcatConv(SYCLEngine* engine, const ConcatConvDesc& desc);
  }
#endif

  namespace xe2
  {
    Ref<InputProcess> newSYCLInputProcess(SYCLEngine* engine, const InputProcessDesc& desc);
    Ref<OutputProcess> newSYCLOutputProcess(SYCLEngine* engine, const OutputProcessDesc& desc);
    Ref<Conv> newSYCLConv(SYCLEngine* engine, const ConvDesc& desc);
    Ref<ConcatConv> newSYCLConcatConv(SYCLEngine* engine, const ConcatConvDesc& desc);
  }

OIDN_NAMESPACE_END
