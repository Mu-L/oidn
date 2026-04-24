// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "math.h"

OIDN_NAMESPACE_BEGIN
namespace math {

  template<typename T>
  struct vec2
  {
    T x, y;

    oidn_host_device_inline vec2() {}
    oidn_host_device_inline vec2(T x) : x(x), y(x) {}
    oidn_host_device_inline vec2(T x, T y) : x(x), y(y) {}

    template<typename U>
    oidn_host_device_inline vec2(vec2<U> v) : x(v.x), y(v.y) {}

  #if defined(OIDN_COMPILE_SYCL)
    template<typename U = T, enable_if_t<esimd::detail::is_simd_type_v<U>, bool> = true>
    oidn_inline void merge(const vec2& val, simd_mask<U::length> mask)
    {
      x.merge(val.x, mask);
      y.merge(val.y, mask);
    }

    template<typename U = T, enable_if_t<esimd::detail::is_simd_type_v<U>, bool> = true>
    oidn_inline void merge(typename U::element_type val, simd_mask<U::length> mask)
    {
      x.merge(val, mask);
      y.merge(val, mask);
    }
  #endif
  };

  template<typename T>
  struct vec3
  {
    T x, y, z;

    oidn_host_device_inline vec3() {}
    oidn_host_device_inline vec3(T x) : x(x), y(x), z(x) {}
    oidn_host_device_inline vec3(T x, T y, T z) : x(x), y(y), z(z) {}

    template<typename U>
    oidn_host_device_inline vec3(vec3<U> v) : x(v.x), y(v.y), z(v.z) {}

  #if defined(OIDN_COMPILE_SYCL)
    template<typename U = T, enable_if_t<esimd::detail::is_simd_type_v<U>, bool> = true>
    oidn_inline void merge(const vec3& val, simd_mask<U::length> mask)
    {
      x.merge(val.x, mask);
      y.merge(val.y, mask);
      z.merge(val.z, mask);
    }

    template<typename U = T, enable_if_t<esimd::detail::is_simd_type_v<U>, bool> = true>
    oidn_inline void merge(typename U::element_type val, simd_mask<U::length> mask)
    {
      x.merge(val, mask);
      y.merge(val, mask);
      z.merge(val, mask);
    }
  #endif
  };

  using vec2f = vec2<float>;
  using vec2i = vec2<int>;
  using vec3f = vec3<float>;
  using vec3i = vec3<int>;

  #define define_vec_binary_op(name, op)                         \
    template<typename T>                                         \
    oidn_host_device_inline vec2<T> name(vec2<T> a, vec2<T> b) { \
      return vec2<T>(a.x op b.x, a.y op b.y);                    \
    }                                                            \
    template<typename T>                                         \
    oidn_host_device_inline vec2<T> name(vec2<T> a, T b) {       \
      return vec2<T>(a.x op b, a.y op b);                        \
    }                                                            \
    template<typename T>                                         \
    oidn_host_device_inline vec2<T> name(T a, vec2<T> b) {       \
      return vec2<T>(a op b.x, a op b.y);                        \
    }                                                            \
    template<typename T>                                         \
    oidn_host_device_inline vec3<T> name(vec3<T> a, vec3<T> b) { \
      return vec3<T>(a.x op b.x, a.y op b.y, a.z op b.z);        \
    }                                                            \
    template<typename T>                                         \
    oidn_host_device_inline vec3<T> name(vec3<T> a, T b) {       \
      return vec3<T>(a.x op b, a.y op b, a.z op b);              \
    }                                                            \
    template<typename T>                                         \
    oidn_host_device_inline vec3<T> name(T a, vec3<T> b) {       \
      return vec3<T>(a op b.x, a op b.y, a op b.z);              \
    }

  define_vec_binary_op(operator+, +)
  define_vec_binary_op(operator-, -)
  define_vec_binary_op(operator*, *)
  define_vec_binary_op(operator/, /)

  #undef define_vec_binary_op

  #define define_vec_unary_func(f)                 \
    template<typename T>                           \
    oidn_host_device_inline vec2<T> f(vec2<T> v) { \
      return vec2<T>(f(v.x), f(v.y));              \
    }                                              \
    template<typename T>                           \
    oidn_host_device_inline vec3<T> f(vec3<T> v) { \
      return vec3<T>(f(v.x), f(v.y), f(v.z));      \
    }

  define_vec_unary_func(log)
  define_vec_unary_func(exp)
  define_vec_unary_func(nan_to_zero)

  #undef define_vec_unary_func

  #define define_vec_binary_func(f)                           \
    template<typename T>                                      \
    oidn_host_device_inline vec2<T> f(vec2<T> a, vec2<T> b) { \
      return vec2<T>(f(a.x, b.x), f(a.y, b.y));               \
    }                                                         \
    template<typename T>                                      \
    oidn_host_device_inline vec2<T> f(vec2<T> a, T b) {       \
      return vec2<T>(f(a.x, b), f(a.y, b));                   \
    }                                                         \
    template<typename T>                                      \
    oidn_host_device_inline vec2<T> f(T a, vec2<T> b) {       \
      return vec2<T>(f(a, b.x), f(a, b.y));                   \
    }                                                         \
    template<typename T>                                      \
    oidn_host_device_inline vec3<T> f(vec3<T> a, vec3<T> b) { \
      return vec3<T>(f(a.x, b.x), f(a.y, b.y), f(a.z, b.z));  \
    }                                                         \
    template<typename T>                                      \
    oidn_host_device_inline vec3<T> f(vec3<T> a, T b) {       \
      return vec3<T>(f(a.x, b), f(a.y, b), f(a.z, b));        \
    }                                                         \
    template<typename T>                                      \
    oidn_host_device_inline vec3<T> f(T a, vec3<T> b) {       \
      return vec3<T>(f(a, b.x), f(a, b.y), f(a, b.z));        \
    }

  define_vec_binary_func(min)
  define_vec_binary_func(max)

  #undef define_vec_binary_func

  #define define_vec_reduce(f)                        \
    template<typename T>                              \
    oidn_host_device_inline T reduce_##f(vec2<T> v) { \
      return f(v.x, v.y);                             \
    }                                                 \
    template<typename T>                              \
    oidn_host_device_inline T reduce_##f(vec3<T> v) { \
      return f(f(v.x, v.y), v.z);                     \
    }

  define_vec_reduce(min)
  define_vec_reduce(max)

  #undef define_vec_reduce

  template<typename T>
  oidn_host_device_inline vec2<T> clamp(vec2<T> v, T minVal, T maxVal)
  {
    return vec2<T>(clamp(v.x, minVal, maxVal),
                   clamp(v.y, minVal, maxVal));
  }

  template<typename T>
  oidn_host_device_inline vec3<T> clamp(vec3<T> v, T minVal, T maxVal)
  {
    return vec3<T>(clamp(v.x, minVal, maxVal),
                   clamp(v.y, minVal, maxVal),
                   clamp(v.z, minVal, maxVal));
  }

#if defined(OIDN_COMPILE_SYCL)
  #define define_vec_simd_binary_op(name, op)                                   \
    template<typename T, int N>                                                 \
    oidn_inline vec2<simd<T, N>> name(vec2<simd<T, N>> a, vec2<simd<T, N>> b) { \
      return vec2<simd<T, N>>(a.x op b.x, a.y op b.y);                          \
    }                                                                           \
    template<typename T, int N>                                                 \
    oidn_inline vec2<simd<T, N>> name(vec2<simd<T, N>> a, T b) {                \
      return vec2<simd<T, N>>(a.x op b, a.y op b);                              \
    }                                                                           \
    template<typename T, int N>                                                 \
    oidn_inline vec2<simd<T, N>> name(T a, vec2<simd<T, N>> b) {                \
      return vec2<simd<T, N>>(a op b.x, a op b.y);                              \
    }                                                                           \
    template<typename T, int N>                                                 \
    oidn_inline vec3<simd<T, N>> name(vec3<simd<T, N>> a, vec3<simd<T, N>> b) { \
      return {a.x op b.x, a.y op b.y, a.z op b.z};                              \
    }                                                                           \
    template<typename T, int N>                                                 \
    oidn_inline vec3<simd<T, N>> name(vec3<simd<T, N>> a, T b) {                \
      return {a.x op b, a.y op b, a.z op b};                                    \
    }                                                                           \
    template<typename T, int N>                                                 \
    oidn_inline vec3<simd<T, N>> name(T a, vec3<simd<T, N>> b) {                \
      return {a op b.x, a op b.y, a op b.z};                                    \
    }

  define_vec_simd_binary_op(operator+, +)
  define_vec_simd_binary_op(operator-, -)
  define_vec_simd_binary_op(operator*, *)
  define_vec_simd_binary_op(operator/, /)

  #undef define_vec_simd_binary_op

  #define define_vec_simd_binary_func(f)                                     \
    template<typename T, int N>                                              \
    oidn_inline vec2<simd<T, N>> f(vec2<simd<T, N>> a, vec2<simd<T, N>> b) { \
      return vec2<simd<T, N>>(f(a.x, b.x), f(a.y, b.y));                     \
    }                                                                        \
    template<typename T, int N>                                              \
    oidn_inline vec2<simd<T, N>> f(vec2<simd<T, N>> a, T b) {                \
      return vec2<simd<T, N>>(f(a.x, b), f(a.y, b));                         \
    }                                                                        \
    template<typename T, int N>                                              \
    oidn_inline vec2<simd<T, N>> f(T a, vec2<simd<T, N>> b) {                \
      return vec2<simd<T, N>>(f(a, b.x), f(a, b.y));                         \
    }                                                                        \
    template<typename T, int N>                                              \
    oidn_inline vec3<simd<T, N>> f(vec3<simd<T, N>> a, vec3<simd<T, N>> b) { \
      return {f(a.x, b.x), f(a.y, b.y), f(a.z, b.z)};                        \
    }                                                                        \
    template<typename T, int N>                                              \
    oidn_inline vec3<simd<T, N>> f(vec3<simd<T, N>> a, T b) {                \
      return {f(a.x, b), f(a.y, b), f(a.z, b)};                              \
    }                                                                        \
    template<typename T, int N>                                              \
    oidn_inline vec3<simd<T, N>> f(T a, vec3<simd<T, N>> b) {                \
      return {f(a, b.x), f(a, b.y), f(a, b.z)};                              \
    }

  define_vec_simd_binary_func(min)
  define_vec_simd_binary_func(max)

  #undef define_vec_simd_binary_func

  template<typename T, int N>
  oidn_inline vec2<simd<T, N>> clamp(vec2<simd<T, N>> v, T minVal, T maxVal)
  {
    return {clamp(v.x, minVal, maxVal),
            clamp(v.y, minVal, maxVal)};
  }

  template<typename T, int N>
  oidn_inline vec3<simd<T, N>> clamp(vec3<simd<T, N>> v, T minVal, T maxVal)
  {
    return {clamp(v.x, minVal, maxVal),
            clamp(v.y, minVal, maxVal),
            clamp(v.z, minVal, maxVal)};
  }
#endif

} // namespace math

using math::vec2;
using math::vec2f;
using math::vec2i;
using math::vec3;
using math::vec3f;
using math::vec3i;

OIDN_NAMESPACE_END