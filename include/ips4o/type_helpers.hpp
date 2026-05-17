#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <hwy/base.h>
#include "KVPair.hpp" 

namespace ss_sort {
namespace detail {

template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
struct is_kvpair : std::false_type {};

template <typename K, typename V>
struct is_kvpair<ss_sort::KVPair<K, V>> : std::true_type {};

template <typename T>
inline constexpr bool is_kvpair_v = is_kvpair<remove_cvref_t<T>>::value;

template <typename T, bool second_is_key = false>
struct ElementKey {
  using type = remove_cvref_t<T>;
};

template <typename K, typename V>
struct ElementKey<ss_sort::KVPair<K, V>, false> {
  using type = K;
};

template <typename K, typename V>
struct ElementKey<ss_sort::KVPair<K, V>, true> {
  using type = V;
};

template <typename T>
using SafePivotKeyT = typename ElementKey<remove_cvref_t<T>, false>::type;

template <typename T, bool second_is_key = false>
using ElementKeyT = typename ElementKey<remove_cvref_t<T>, second_is_key>::type;

template <typename T>
inline constexpr bool is_safe_trivially_copyable_v = std::is_trivially_copyable_v<T>;

template <typename T>
inline constexpr bool is_hwy_vqsort_supported_v =
    std::is_same_v<remove_cvref_t<T>, uint16_t> ||
    std::is_same_v<remove_cvref_t<T>, uint32_t> ||
    std::is_same_v<remove_cvref_t<T>, uint64_t> ||
    std::is_same_v<remove_cvref_t<T>, int16_t> ||
    std::is_same_v<remove_cvref_t<T>, int32_t> ||
    std::is_same_v<remove_cvref_t<T>, int64_t> ||
    std::is_same_v<remove_cvref_t<T>, float> ||
    std::is_same_v<remove_cvref_t<T>, double> ||
    std::is_same_v<remove_cvref_t<T>, hwy::uint128_t> ||
    std::is_same_v<remove_cvref_t<T>, hwy::K32V32> ||
    std::is_same_v<remove_cvref_t<T>, hwy::K64V64>;

template <typename T>
inline constexpr bool is_hwy_vqsort_pair_wrapper_v =
    std::is_same_v<remove_cvref_t<T>, hwy::K32V32> ||
    std::is_same_v<remove_cvref_t<T>, hwy::K64V64>;

template <typename T>
inline constexpr bool is_hwy_vqsort_supported_scalar_v =
    is_hwy_vqsort_supported_v<T> && !is_hwy_vqsort_pair_wrapper_v<T>;

template <typename T>
inline constexpr bool is_sample_sort_scalar_v =
    !is_kvpair_v<T> &&
    is_hwy_vqsort_supported_scalar_v<T>;

template <typename T>
inline constexpr bool is_simd_vectorizable_scalar_v =
    std::is_arithmetic_v<remove_cvref_t<T>>;

template <typename T>
struct GetKVType { using K = void; using V = void; };

template <typename _K, typename _V>
struct GetKVType<ss_sort::KVPair<_K, _V>> {
  using K = _K;
  using V = _V;
};

template <bool second_is_key = false, typename T>
inline decltype(auto) key_of(const T& value) {
  if constexpr (is_kvpair_v<T>) {
    if constexpr (second_is_key) {
      return (value.second);
    } else {
      return (value.first);
    }
  } else {
    return (value);
  }
}

template <typename K, bool second_is_key = false, typename P>
inline K pivot_key_as(const P& p) {
  return static_cast<K>(key_of<second_is_key>(p));
}

}  // namespace detail
}  // namespace ss_sort
