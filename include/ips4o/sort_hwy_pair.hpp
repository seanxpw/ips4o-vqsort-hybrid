#pragma once

#include <hwy/highway.h>
#include <hwy/base.h>
#include <hwy/contrib/sort/vqsort.h>

#include <parlay/slice.h>

#include <type_traits>
#include <iterator>
#include <utility>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <limits>
#include <memory>   // addressof
#include <type_traits>
#include <cstddef>
#include <cstdint>

#include "type_helpers.hpp"
#include "KVPair.hpp"

namespace hn = hwy::HWY_NAMESPACE;

namespace parlay {
namespace internal_simd {


template <typename To, typename From>
static inline To bit_cast_memcpy(const From& src) {
  static_assert(sizeof(To) == sizeof(From), "bit_cast size mismatch");
  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename K>
static inline hwy::MakeUnsigned<K> to_radix_key_scalar(K v) {
  using U = hwy::MakeUnsigned<K>;
  U bits = bit_cast_memcpy<U>(v);

  if constexpr (std::is_floating_point_v<K>) {
    constexpr int shift = int(sizeof(K) * 8 - 1);
    const U msb = U(1) << shift;
    using SU = std::make_signed_t<U>;
    const U sign = bits >> shift;                 // 0/1
    const U all_ones_if_neg = U(-SU(sign));       // 0/~0
    const U mask = all_ones_if_neg | msb;         // neg:~0, pos:msb
    bits ^= mask;
  } else if constexpr (std::is_signed_v<K>) {
    bits ^= (U(1) << (sizeof(K) * 8 - 1));
  }
  return bits;
}

template <typename K>
static inline K from_radix_key_scalar(hwy::MakeUnsigned<K> key) {
  using U = hwy::MakeUnsigned<K>;
  if constexpr (std::is_floating_point_v<K>) {
    constexpr int shift = int(sizeof(K) * 8 - 1);
    const U msb = U(1) << shift;
    const U mask = ((key >> shift) - 1) | msb;
    key ^= mask;
  } else if constexpr (std::is_signed_v<K>) {
    key ^= (U(1) << (sizeof(K) * 8 - 1));
  }
  return bit_cast_memcpy<K>(key);
}

template <typename K, class D, class V>
HWY_INLINE V to_radix_key_vec_bits(D du, V bits_u) {
  using U = hn::TFromD<D>;
  const auto msb = hn::Set(du, U(1) << (sizeof(U) * 8 - 1));

  if constexpr (std::is_floating_point_v<K>) {
    const auto zero = hn::Zero(du);
    const auto sign_set = hn::Ne(hn::And(bits_u, msb), zero);
    const auto all_ones = hn::Set(du, static_cast<U>(~U(0)));
    const auto mask = hn::IfThenElse(sign_set, all_ones, msb);
    return hn::Xor(bits_u, mask);
  } else if constexpr (std::is_signed_v<K>) {
    return hn::Xor(bits_u, msb);
  } else {
    return bits_u;
  }
}
// [新增] 向量版的反向变换：从 RadixKey 恢复原始 Key
template <typename K, class D, class V>
HWY_INLINE V from_radix_key_vec_bits(D du, V bits_u) {
  using U = hn::TFromD<D>;
  const auto msb = hn::Set(du, U(1) << (sizeof(U) * 8 - 1));

  if constexpr (std::is_floating_point_v<K>) {
    // 标量逻辑: mask = ((key >> shift) - 1) | msb; key ^= mask;
    // 向量实现:
    const auto one = hn::Set(du, 1);
    // 1. Shift Right (Logical)
    // 注意：ShiftRight 需要编译期常量，或者用 ShiftRightSame
    auto shifted = hn::ShiftRight<sizeof(U) * 8 - 1>(bits_u);
    // 2. Sub 1
    auto sub = hn::Sub(shifted, one);
    // 3. Or MSB
    auto mask = hn::Or(sub, msb);
    // 4. Xor
    return hn::Xor(bits_u, mask);
  } else if constexpr (std::is_signed_v<K>) {
    // 标量逻辑: key ^= msb
    return hn::Xor(bits_u, msb);
  } else {
    return bits_u;
  }
}
// unstable begins here

// ---------------------------------------------------------
// Pass 1: [K, V] -> Transform K -> Swap -> [V, RadixK]
// ---------------------------------------------------------
template <typename K, typename V>
void transform_forward_and_swap_hwy(KVPair<K, V>* data, size_t n) {
    using U = hwy::MakeUnsigned<K>; // 将 K 视为同大小的无符号数处理
    const hn::ScalableTag<U> d;
    using Vec = hn::Vec<decltype(d)>; 
    auto* ptr = reinterpret_cast<U*>(data);
    size_t i = 0;
    size_t lanes = hn::Lanes(d);

    // SIMD Loop
    // 每次处理 'lanes' 对数据
    for (; i + lanes <= n; i += lanes) {
        Vec v_k, v_v;
        // 1. LoadInterleaved2: 自动将 [K,V,K,V...] 分离到 v_k 和 v_v 寄存器
        hn::LoadInterleaved2(d, ptr + 2 * i, v_k, v_v);

        // 2. Transform K -> RadixKey
        auto v_radix_k = to_radix_key_vec_bits<K>(d, v_k);

        // 3. StoreInterleaved2: 交换顺序写入 (先 V, 后 RadixK)
        // 内存变为: [V, RadixK, V, RadixK ...]
        hn::StoreInterleaved2(v_v, v_radix_k, d, ptr + 2 * i);
    }

    // Scalar Fallback (处理剩余元素)
    for (; i < n; ++i) {
        // 读取
        K k = data[i].first;
        V v = data[i].second;
        // 变换
        U rk = to_radix_key_scalar<K>(k);
        // 交换写入 (强转指针以写入无符号数)
        // first (low addr) <- v
        // second (high addr) <- rk
        *reinterpret_cast<V*>(&data[i].first) = v;
        *reinterpret_cast<U*>(&data[i].second) = rk;
    }
}

// ---------------------------------------------------------
// Pass 2: [V, RadixK] -> Untransform K -> Swap -> [K, V]
// ---------------------------------------------------------
template <typename K, typename V>
void restore_backward_and_swap_hwy(KVPair<K, V>* data, size_t n) {
    using U = hwy::MakeUnsigned<K>;
    const hn::ScalableTag<U> d; // Descriptor
    
    // ✅ 修复 1: 定义向量类型
    using Vec = hn::Vec<decltype(d)>; 

    auto* ptr = reinterpret_cast<U*>(data);
    size_t i = 0;
    size_t lanes = hn::Lanes(d);

    for (; i + lanes <= n; i += lanes) {
        // ✅ 修复 2: 声明变量为向量 (Vector)，而不是标量 (Scalar)
        Vec v_v, v_radix_k; 

        // 1. LoadInterleaved2: 读取当前的 [V, RadixK]
        // 现在 v_v 和 v_radix_k 是向量，类型匹配了
        hn::LoadInterleaved2(d, ptr + 2 * i, v_v, v_radix_k);

        // 2. Untransform RadixKey -> K
        // from_radix_key_vec_bits 需要接收向量，返回向量
        auto v_k = from_radix_key_vec_bits<K>(d, v_radix_k);

        // 3. StoreInterleaved2: 交换顺序写入 (先 K, 后 V)
        // 参数顺序检查: StoreInterleaved2(v0, v1, d, ptr) -> 正确
        hn::StoreInterleaved2(v_k, v_v, d, ptr + 2 * i);
    }

    // Scalar Fallback (保持不变，逻辑是正确的)
    for (; i < n; ++i) {
        // 当前内存布局是 [V, RadixK]
        // data[i].first 的物理位置存的是 V
        V v = *reinterpret_cast<V*>(&data[i].first);
        // data[i].second 的物理位置存的是 RadixK
        U rk = *reinterpret_cast<U*>(&data[i].second);
        
        K k = from_radix_key_scalar<K>(rk);
        
        // 归位: 写入 [K, V]
        data[i].first = k;
        data[i].second = v;
    }
}

template <typename K, typename V>
void unstable_sort_pairs_hwy(KVPair<K, V>* in,
                             size_t n,
                             bool ascending = true) {
    // 1. 静态断言
    static_assert(sizeof(K) == sizeof(V), "K/V size mismatch");
    static_assert(sizeof(K) == 4 || sizeof(K) == 8, "Only 32/64-bit supported");
    // 对齐检查
    if constexpr (sizeof(K) == 8) {
        assert((reinterpret_cast<uintptr_t>(in) % 16) == 0);
    } else {
        assert((reinterpret_cast<uintptr_t>(in) % 8) == 0);
    }

    if (n <= 1) return;

    // 2. 变换布局与 Key 编码
    // [K, V] -> [V, RadixKey]
    // 此时内存里是一个 Little Endian 的大整数，RadixKey 在高位，Value 在低位
    internal_simd::transform_forward_and_swap_hwy(in, n);

    // 3. 排序
    // 直接调用 VQSort，将内存视为 hwy::K64V64 (128-bit) 或 hwy::K32V32 (64-bit) 数组
    // 因为 Key 在高位，普通的大整数比较逻辑正好满足 "先比Key，后比Value" 的需求
    if constexpr (sizeof(K) == 4) {
        auto* ptr = reinterpret_cast<hwy::K32V32*>(in);
        if (ascending) {
            hwy::VQSort(ptr, n, hwy::SortAscending());
        } else {
            hwy::VQSort(ptr, n, hwy::SortDescending());
        }
    } else {
        auto* ptr = reinterpret_cast<hwy::K64V64*>(in);
        if (ascending) {
            hwy::VQSort(ptr, n, hwy::SortAscending());
        } else {
            hwy::VQSort(ptr, n, hwy::SortDescending());
        }
    }

    // 4. 还原布局与 Key 解码
    // [V, RadixKey] -> [K, V]
    internal_simd::restore_backward_and_swap_hwy(in, n);
}

// Slice Wrapper
template <typename InIterator>
void unstable_sort_pairs_hwy(parlay::slice<InIterator, InIterator> In,
                             bool ascending = true) {
    using T = std::remove_cv_t<typename std::iterator_traits<InIterator>::value_type>;
    static_assert(is_kvpair_v<T>, "Must be KVPair");
    using K = typename GetKVType<T>::K;
    using V = typename GetKVType<T>::V;
    
    unstable_sort_pairs_hwy<K, V>(In.begin(), In.size(), ascending);
}
} // namespace internal
} // namespace parlay
