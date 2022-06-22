// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../conv.h"
#include "sycl_device.h"

namespace oidn {

  class SYCLConv : public Conv
  {
  public:
    SYCLConv(const Ref<SYCLDevice>& device, const ConvDesc& desc);
    void run() override;

  private:
    Ref<SYCLDevice> device;
  };

} // namespace oidn