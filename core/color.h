// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vec.h"

OIDN_NAMESPACE_BEGIN

  struct TransferFunction
  {
    enum class Type
    {
      Linear,
      SRGB,
      PU,
      Log,
    };

    static constexpr oidn_constant float yMax = 65504.f; // maximum HDR value

    Type type;
    const oidn_global float* inputScalePtr = nullptr;
    float inputScale   = 1.f;
    float outputScale  = 1.f;
    float normScale    = 1.f;
    float rcpNormScale = 1.f;

    struct SRGB
    {
      static constexpr oidn_constant float a  =  12.92f;
      static constexpr oidn_constant float b  =  1.055f;
      static constexpr oidn_constant float c  =  1.f/2.4f;
      static constexpr oidn_constant float d  = -0.055f;
      static constexpr oidn_constant float y0 =  0.0031308f;
      static constexpr oidn_constant float x0 =  0.04045f;

      static oidn_host_device_inline float forward(float y)
      {
        if (y <= y0)
          return a * y;
        else
          return b * math::pow(y, c) + d;
      }

      static oidn_host_device_inline float inverse(float x)
      {
        if (x <= x0)
          return x / a;
        else
          return math::pow((x - d) / b, 1.f/c);
      }

    #if defined(OIDN_COMPILE_SYCL)
      template<int N>
      static oidn_inline simd<float, N> forward(simd<float, N> y)
      {
        simd<float, N> v0 = a * y;
        simd<float, N> v  = b * pow(y, c) + d;
        v.merge(v0, y <= y0);
        return v;
      }

      template<int N>
      static oidn_inline simd<float, N> inverse(simd<float, N> x)
      {
        simd<float, N> v0 = x / a;
        simd<float, N> v  = pow((x - d) / b, 1.f/c);
        v.merge(v0, x <= x0);
        return v;
      }
    #endif
    };

    struct PU
    {
      static constexpr oidn_constant float a  =  1.41283765e+03f;
      static constexpr oidn_constant float b  =  1.64593172e+00f;
      static constexpr oidn_constant float c  =  4.31384981e-01f;
      static constexpr oidn_constant float d  = -2.94139609e-03f;
      static constexpr oidn_constant float e  =  1.92653254e-01f;
      static constexpr oidn_constant float f  =  6.26026094e-03f;
      static constexpr oidn_constant float g  =  9.98620152e-01f;
      static constexpr oidn_constant float y0 =  1.57945760e-06f;
      static constexpr oidn_constant float y1 =  3.22087631e-02f;
      static constexpr oidn_constant float x0 =  2.23151711e-03f;
      static constexpr oidn_constant float x1 =  3.70974749e-01f;

      static oidn_host_device_inline float forward(float y)
      {
        if (y <= y0)
          return a * y;
        else if (y <= y1)
          return b * math::pow(y, c) + d;
        else
          return e * math::log(y + f) + g;
      }

      static oidn_host_device_inline float inverse(float x)
      {
        if (x <= x0)
          return x / a;
        else if (x <= x1)
          return math::pow((x - d) / b, 1.f/c);
        else
          return math::exp((x - g) / e) - f;
      }

    #if defined(OIDN_COMPILE_SYCL)
      template<int N>
      static oidn_inline simd<float, N> forward(simd<float, N> y)
      {
        simd<float, N> v0 = a * y;
        simd<float, N> v1 = b * pow(y, c) + d;
        simd<float, N> v  = e * log(y + f) + g;
        v.merge(v1, y <= y1);
        v.merge(v0, y <= y0);
        return v;
      }

      template<int N>
      static oidn_inline simd<float, N> inverse(simd<float, N> x)
      {
        simd<float, N> v0 = x / a;
        simd<float, N> v1 = pow((x - d) / b, 1.f/c);
        simd<float, N> v  = exp((x - g) / e) - f;
        v.merge(v1, x <= x1);
        v.merge(v0, x <= x0);
        return v;
      }
    #endif
    };

  #if !defined(OIDN_COMPILE_METAL_DEVICE)
    explicit TransferFunction(Type type = Type::Linear);

    Type getType() const { return type; }

    void setInputScale(float inputScale)
    {
      this->inputScalePtr = nullptr;
      this->inputScale  = inputScale;
      this->outputScale = (inputScale != 0.f) ? (1.f / inputScale) : 0.f;
    }

    void setInputScale(const float* inputScalePtr)
    {
      this->inputScalePtr = inputScalePtr;
      this->inputScale  = 1.f;
      this->outputScale = 1.f;
    }
  #endif

    oidn_host_device_inline float getInputScale() const
    {
      return inputScalePtr ? *inputScalePtr : inputScale;
    }

    oidn_host_device_inline float getOutputScale() const
    {
      if (inputScalePtr)
      {
        const float inputScale = *inputScalePtr;
        return (inputScale != 0.f) ? (1.f / inputScale) : 0.f;
      }
      return outputScale;
    }

    template<typename T>
    oidn_host_device_inline vec3<T> forward(vec3<T> y) const
    {
      switch (type)
      {
      case Type::Linear:
        return y;

      case Type::SRGB:
        return vec3<T>(SRGB::forward(y.x), SRGB::forward(y.y), SRGB::forward(y.z));

      case Type::PU:
        return vec3<T>(PU::forward(y.x), PU::forward(y.y), PU::forward(y.z)) * normScale;

      case Type::Log:
        return math::log(y + 1.f) * normScale;

      default:
        return T(0.f);
      }
    }

    template<typename T>
    oidn_host_device_inline vec3<T> inverse(vec3<T> x) const
    {
      switch (type)
      {
      case Type::Linear:
        return x;

      case Type::SRGB:
        return vec3<T>(SRGB::inverse(x.x), SRGB::inverse(x.y), SRGB::inverse(x.z));

      case Type::PU:
        return vec3<T>(PU::inverse(x.x * rcpNormScale), PU::inverse(x.y * rcpNormScale), PU::inverse(x.z * rcpNormScale));

      case Type::Log:
        return math::exp(x * rcpNormScale) - 1.f;

      default:
        return T(0.f);
      }
    }
  };

  // Computes the luminance of an RGB color
  oidn_host_device_inline float luminance(vec3f c)
  {
    return 0.212671f * c.x + 0.715160f * c.y + 0.072169f * c.z;
  }

OIDN_NAMESPACE_END
