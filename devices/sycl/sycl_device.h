// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/device.h"
#include <level_zero/ze_api.h>

OIDN_NAMESPACE_BEGIN

  class SYCLEngine;

  // GPU architecture
  enum class SYCLArch
  {
    Unknown = 0,
    Gen9    = 1,
    XeHPG   = 1000,
    XeHPC   = 2000,
  };

  class SYCLPhysicalDevice : public PhysicalDevice
  {
  public:
    sycl::device syclDevice;

    SYCLPhysicalDevice(const sycl::device& syclDevice, int score);
  };

  class SYCLDevice : public SYCLDeviceBase
  { 
  public:
    static std::vector<Ref<PhysicalDevice>> getPhysicalDevices();
    static SYCLArch getArch(const sycl::device& syclDevice);
    static int getScore(const sycl::device& syclDevice);

    SYCLDevice(const std::vector<sycl::queue>& syclQueues = {});
    explicit SYCLDevice(const Ref<SYCLPhysicalDevice>& physicalDevice);

    DeviceType getType() const override { return DeviceType::SYCL; }
    ze_context_handle_t getZeContext() const { return zeContext; }
    
    Engine* getEngine(int i) const override { return (Engine*)engines[i].get(); }
    int getNumEngines() const override { return int(engines.size()); }
    
    int getInt(const std::string& name) override;
    void setInt(const std::string& name, int value) override;

    Storage getPointerStorage(const void* ptr) override;

    void submitBarrier() override;
    void wait() override;
    
    // Manually sets the dependent events for the next command on all engines
    void setDepEvents(const std::vector<sycl::event>& events);
    void setDepEvents(const sycl::event* events, int numEvents) override;
    
    // Gets the list of events corresponding to the completion of all commands
    std::vector<sycl::event> getDoneEvents();
    void getDoneEvent(sycl::event& event) override;

    SYCLArch getArch() const { return arch; }

  private:
    void preinit();
    void init() override;

    sycl::context syclContext;
    ze_context_handle_t zeContext = nullptr; // Level Zero context
    std::vector<Ref<SYCLEngine>> engines;
    SYCLArch arch = SYCLArch::Unknown;
    int numSubdevices = 0; // autodetect by default

    // Used only for initialization
    Ref<SYCLPhysicalDevice> physicalDevice;
    std::vector<sycl::queue> syclQueues;
  };

OIDN_NAMESPACE_END