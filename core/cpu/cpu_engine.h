// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../engine.h"
#include "cpu_device.h"

namespace oidn {

  class CPUEngine : public Engine
  { 
  public:
    explicit CPUEngine(const Ref<CPUDevice>& device);

    Device* getDevice() const override { return device; }

    // Ops
    std::shared_ptr<Upsample> newUpsample(const UpsampleDesc& desc) override;
    std::shared_ptr<Autoexposure> newAutoexposure(const ImageDesc& srcDesc) override;
    std::shared_ptr<InputProcess> newInputProcess(const InputProcessDesc& desc) override;
    std::shared_ptr<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) override;
    std::shared_ptr<ImageCopy> newImageCopy() override;

    // Runs a parallel host task in the thread arena (if it exists)
    void runHostTask(std::function<void()>&& f) override;

    // Enqueues a host function
    void submitHostFunc(std::function<void()>&& f) override;

    void wait() override {}

  protected:
    CPUDevice* device;
  };

} // namespace oidn