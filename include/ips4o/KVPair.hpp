#pragma once
#include <cstddef>
#include <type_traits>

namespace ss_sort {

#pragma pack(push, 1)

template <class K, class V>
struct alignas(sizeof(K) + sizeof(V)) KVPair {
  using first_type = K;
  using second_type = V;

  static constexpr size_t kSizeK = sizeof(K);
  static constexpr size_t kSizeV = sizeof(V);
  static constexpr size_t kAlign = kSizeK + kSizeV;

  static_assert(kSizeK == kSizeV,
                "KVPair only requires equal K/V byte width, not identical K/V types.");

  static_assert((kAlign & (kAlign - 1)) == 0,
                "KVPair requires sizeof(K)+sizeof(V) to be power-of-two.");

  static_assert(kAlign == 8 || kAlign == 16,
                "KVPair only supports 8-byte or 16-byte element size.");

  K first;   // key
  V second;  // value

  // ============================================================
  // 1. Pair 之间的比较 (通常用于完全排序)
  // ============================================================
  
  [[nodiscard]] friend bool operator<(const KVPair& a, const KVPair& b) noexcept {
    return a.first < b.first;
  }
  [[nodiscard]] friend bool operator>(const KVPair& a, const KVPair& b) noexcept {
    return a.first > b.first;
  }
  [[nodiscard]] friend bool operator<=(const KVPair& a, const KVPair& b) noexcept {
    return a.first <= b.first;
  }
  [[nodiscard]] friend bool operator>=(const KVPair& a, const KVPair& b) noexcept {
    return a.first >= b.first;
  }
  [[nodiscard]] friend bool operator==(const KVPair& a, const KVPair& b) noexcept {
    return a.first == b.first;
  }
  [[nodiscard]] friend bool operator!=(const KVPair& a, const KVPair& b) noexcept {
    return a.first != b.first;
  }

  // ============================================================
  // 2. Pair 与 Key (Scalar) 的混合比较 (解决你的报错)
  // ============================================================

  // Case A: KVPair < Key
  [[nodiscard]] friend bool operator<(const KVPair& p, const K& k) noexcept {
    return p.first < k;
  }
  [[nodiscard]] friend bool operator>(const KVPair& p, const K& k) noexcept {
    return p.first > k;
  }
  [[nodiscard]] friend bool operator<=(const KVPair& p, const K& k) noexcept {
    return p.first <= k;
  }
  [[nodiscard]] friend bool operator>=(const KVPair& p, const K& k) noexcept {
    return p.first >= k;
  }
  [[nodiscard]] friend bool operator==(const KVPair& p, const K& k) noexcept {
    return p.first == k;
  }
  [[nodiscard]] friend bool operator!=(const KVPair& p, const K& k) noexcept {
    return p.first != k;
  }

  // Case B: Key < KVPair (反向比较，比如 pivot < elem)
  [[nodiscard]] friend bool operator<(const K& k, const KVPair& p) noexcept {
    return k < p.first;
  }
  [[nodiscard]] friend bool operator>(const K& k, const KVPair& p) noexcept {
    return k > p.first;
  }
  [[nodiscard]] friend bool operator<=(const K& k, const KVPair& p) noexcept {
    return k <= p.first;
  }
  [[nodiscard]] friend bool operator>=(const K& k, const KVPair& p) noexcept {
    return k >= p.first;
  }
  [[nodiscard]] friend bool operator==(const K& k, const KVPair& p) noexcept {
    return k == p.first;
  }
  [[nodiscard]] friend bool operator!=(const K& k, const KVPair& p) noexcept {
    return k != p.first;
  }
};

#pragma pack(pop)

} // namespace ss_sort
