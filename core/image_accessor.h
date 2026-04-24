// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vec.h"

// ISPC forward declarations
namespace ispc
{
  struct ImageAccessor;
}

OIDN_NAMESPACE_BEGIN

  struct ImageAccessor
  {
    oidn_global char* ptr;
    size_t hByteStride; // row stride in number of bytes
    size_t wByteStride; // pixel stride in number of bytes
    DataType dataType;  // data type
    int C, H, W;        // channels (1-3), height, width

    oidn_host_device_inline size_t getByteOffset(int h, int w) const
    {
      return size_t(h) * hByteStride + size_t(w) * wByteStride;
    }

  #if defined(OIDN_COMPILE_SYCL)
    template<int N>
    oidn_inline simd<uint32_t, N> getByteOffset(int h, simd<int, N> w) const
    {
      return uint32_t(h) * uint32_t(hByteStride) + simd<uint32_t, N>(w) * uint32_t(wByteStride);
    }
  #endif

    template<typename T = float>
    oidn_host_device_inline vec3<T> get3(int h, int w) const
    {
      const oidn_global void* pixelPtr = ptr + getByteOffset(h, w);
      if (dataType == DataType::Float32)
      {
        const oidn_global float* pixel = static_cast<const oidn_global float*>(pixelPtr);
        if (C == 3)
          return vec3<T>(pixel[0], pixel[1], pixel[2]);
        else if (C == 2)
          return vec3<T>(pixel[0], pixel[1], pixel[1]);
        else // if (C == 1)
          return vec3<T>(pixel[0], pixel[0], pixel[0]);
      }
      else // if (dataType == DataType::Float16)
      {
        const oidn_global half* pixel = static_cast<const oidn_global half*>(pixelPtr);
        if (C == 3)
          return vec3<T>(pixel[0], pixel[1], pixel[2]);
        else if (C == 2)
          return vec3<T>(pixel[0], pixel[1], pixel[1]);
        else // if (C == 1)
          return vec3<T>(pixel[0], pixel[0], pixel[0]);
      }
    }

    template<typename T>
    oidn_host_device_inline void set3(int h, int w, vec3<T> value) const
    {
      oidn_global void* pixelPtr = ptr + getByteOffset(h, w);
      if (dataType == DataType::Float32)
      {
        oidn_global float* pixel = static_cast<oidn_global float*>(pixelPtr);
        if (C == 3)
        {
          pixel[0] = value.x;
          pixel[1] = value.y;
          pixel[2] = value.z;
        }
        else if (C == 2)
        {
          pixel[0] = value.x;
          pixel[1] = value.y;
        }
        else // if (C == 1)
          pixel[0] = value.x;
      }
      else // if (dataType == DataType::Float16)
      {
        oidn_global half* pixel = static_cast<oidn_global half*>(pixelPtr);
        if (C == 3)
        {
          pixel[0] = value.x;
          pixel[1] = value.y;
          pixel[2] = value.z;
        }
        else if (C == 2)
        {
          pixel[0] = value.x;
          pixel[1] = value.y;
        }
        else // if (C == 1)
          pixel[0] = value.x;
      }
    }

  #if defined(OIDN_COMPILE_SYCL)
    template<typename T = float, int N>
    oidn_inline vec3<simd<T, N>> get3(int h, simd<int, N> w, simd_mask<N> mask) const
    {
      vec3<simd<T, N>> c;
      const simd<uint32_t, N> offset = getByteOffset(h, w);

      if (dataType == DataType::Float32)
      {
        const float* base = reinterpret_cast<const float*>(ptr);

      #if defined(OIDN_ARCH_XEHPG) || defined(OIDN_ARCH_XEHPC) || defined(OIDN_ARCH_XE2)
        if (C == 3)
        {
          simd<float, N*3> data = lsc_gather<float, 3>(base, offset, mask);
          c.x = data.template select<N, 1>(0);
          c.y = data.template select<N, 1>(N);
          c.z = data.template select<N, 1>(N*2);
        }
        else if (C == 2)
        {
          simd<float, N*2> data = lsc_gather<float, 2>(base, offset, mask);
          c.x = data.template select<N, 1>(0);
          c.y = data.template select<N, 1>(N);
          c.z = c.y;
        }
        else // C == 1
        {
          c.x = lsc_gather<float>(base, offset, mask);
          c.y = c.z = c.x;
        }
      #else
        c.x = gather<float, N>(base, offset, mask);
        if (C >= 2)
        {
          c.y = gather<float, N>(base, offset + uint32_t(sizeof(float)), mask);
          if (C >= 3)
            c.z = gather<float, N>(base, offset + uint32_t(2 * sizeof(float)), mask);
          else
            c.z = c.y;
        }
        else
          c.y = c.z = c.x;
      #endif
      }
      else // Float16
      {
        const half* base = reinterpret_cast<const half*>(ptr);

      #if defined(OIDN_ARCH_XEHPG) || defined(OIDN_ARCH_XEHPC) || defined(OIDN_ARCH_XE2)
        // d16u32 does not support vector lsc_gather (x2/x3)
        c.x = lsc_gather<half>(base, offset, mask);
        if (C >= 2)
        {
          c.y = lsc_gather<half>(base, offset + uint32_t(sizeof(half)), mask);
          if (C >= 3)
            c.z = lsc_gather<half>(base, offset + uint32_t(2 * sizeof(half)), mask);
          else
            c.z = c.y;
        }
        else
          c.y = c.z = c.x;
      #else
        c.x = gather<half, N>(base, offset, mask);
        if (C >= 2)
        {
          c.y = gather<half, N>(base, offset + uint32_t(sizeof(half)), mask);
          if (C >= 3)
            c.z = gather<half, N>(base, offset + uint32_t(2 * sizeof(half)), mask);
          else
            c.z = c.y;
        }
        else
          c.y = c.z = c.x;
      #endif
      }

      return c;
    }

    template<typename T, int N>
    oidn_inline void set3(int h, simd<int, N> w, vec3<simd<T, N>> value, simd_mask<N> mask) const
    {
      const simd<uint32_t, N> offset = getByteOffset(h, w);

      if (dataType == DataType::Float32)
      {
        float* base = reinterpret_cast<float*>(ptr);

      #if defined(OIDN_ARCH_XEHPG) || defined(OIDN_ARCH_XEHPC) || defined(OIDN_ARCH_XE2)
        if (C == 3)
        {
          simd<float, N*3> data;
          data.template select<N, 1>(0)   = simd<float, N>(value.x);
          data.template select<N, 1>(N)   = simd<float, N>(value.y);
          data.template select<N, 1>(N*2) = simd<float, N>(value.z);
          lsc_scatter<float, 3>(base, offset, data, mask);
        }
        else if (C == 2)
        {
          simd<float, N*2> data;
          data.template select<N, 1>(0) = simd<float, N>(value.x);
          data.template select<N, 1>(N) = simd<float, N>(value.y);
          lsc_scatter<float, 2>(base, offset, data, mask);
        }
        else // C == 1
        {
          lsc_scatter<float>(base, offset, simd<float, N>(value.x), mask);
        }
      #else
        scatter<float, N>(base, offset, value.x, mask);
        if (C >= 2)
        {
          scatter<float, N>(base, offset + uint32_t(sizeof(float)), value.y, mask);
          if (C >= 3)
            scatter<float, N>(base, offset + uint32_t(sizeof(float)*2), value.z, mask);
        }
      #endif
      }
      else // Float16
      {
        half* base = reinterpret_cast<half*>(ptr);

      #if defined(OIDN_ARCH_XEHPG) || defined(OIDN_ARCH_XEHPC) || defined(OIDN_ARCH_XE2)
        // d16u32 does not support vector lsc_gather (x2/x3)
        lsc_scatter<half>(base, offset, simd<half, N>(value.x), mask);
        if (C >= 2)
        {
          lsc_scatter<half>(base, offset + uint32_t(sizeof(half)), simd<half, N>(value.y), mask);
          if (C >= 3)
            lsc_scatter<half>(base, offset + uint32_t(sizeof(half)*2), simd<half, N>(value.z), mask);
        }
      #else
        scatter<half, N>(base, offset, simd<half, N>(value.x), mask);
        if (C >= 2)
        {
          scatter<half, N>(base, offset + uint32_t(sizeof(half)), simd<half, N>(value.y), mask);
          if (C >= 3)
            scatter<half, N>(base, offset + uint32_t(sizeof(half)*2), simd<half, N>(value.z), mask);
        }
      #endif
      }
    }
  #endif
  };

OIDN_NAMESPACE_END