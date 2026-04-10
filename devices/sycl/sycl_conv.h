// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/conv.h"
#include "core/concat_conv.h"
#include "sycl_engine.h"

OIDN_NAMESPACE_BEGIN

  namespace xelp {
    Ref<Conv> newSYCLConv(SYCLEngine* engine, const ConvDesc& desc);
    Ref<ConcatConv> newSYCLConcatConv(SYCLEngine* engine, const ConcatConvDesc& desc);
  }

  namespace xehpg {
    Ref<Conv> newSYCLConv(SYCLEngine* engine, const ConvDesc& desc);
    Ref<ConcatConv> newSYCLConcatConv(SYCLEngine* engine, const ConcatConvDesc& desc);
  }

#if defined(__linux__)
  namespace xehpc {
    Ref<Conv> newSYCLConv(SYCLEngine* engine, const ConvDesc& desc);
    Ref<ConcatConv> newSYCLConcatConv(SYCLEngine* engine, const ConcatConvDesc& desc);
  }
#endif

  namespace xe2 {
    Ref<Conv> newSYCLConv(SYCLEngine* engine, const ConvDesc& desc);
    Ref<ConcatConv> newSYCLConcatConv(SYCLEngine* engine, const ConcatConvDesc& desc);
  }

OIDN_NAMESPACE_END
