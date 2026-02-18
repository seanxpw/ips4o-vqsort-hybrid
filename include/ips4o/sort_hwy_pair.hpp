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

#include "../utils/type_helpers.hpp"
#include "../utils/KVPair.hpp"

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
// =========================================================
// SIMD pack: AoS pair -> packed keys with source index
//   K32: packed64[i]  = (radix32<<32) | idx32
//   K64: packed128[i] = {lo=idx64, hi=radix64}
// =========================================================
template <typename K, typename V>
static inline void pack_k32_idx_simd(const parlay::KVPair<K, V>* in,
                                     size_t n,
                                     uint64_t* packed,
                                     bool ascending) {
  using U32 = uint32_t;
  const hn::ScalableTag<U32> du; // 32-bit lanes
  const size_t lanes = hn::Lanes(du);

  size_t i = 0;

#if HWY_TARGET != HWY_SCALAR
  for (; i + lanes <= n; i += lanes) {
    // Load Keys (Stride = 2 words)
    // 假设 layout 是 [K, V, K, V...]
    const U32* src = reinterpret_cast<const U32*>(in + i);
    auto v_key_bits = hn::Zero(du);
    auto v_dummy    = hn::Zero(du);
    hn::LoadInterleaved2(du, src, v_key_bits, v_dummy);

    // Key Transformation
    auto v_rk = to_radix_key_vec_bits<K>(du, v_key_bits);
    if (!ascending) {
      const auto all_ones = hn::Set(du, static_cast<U32>(~U32(0)));
      v_rk = hn::Xor(v_rk, all_ones);
    }

    // Generate Indices
    const auto v_idx = hn::Iota(du, static_cast<U32>(i));

    // ★ 核心优化 ★
    // 原逻辑: packed = (rk << 32) | idx
    // 内存布局(LE): [idx_low, idx_hi(0), rk_low, rk_hi(0)] (错误)
    // 实际布局(LE): uint64 { Low32=Idx, High32=Key }
    // 内存字节:     [Idx Byte0..3] [Key Byte0..3]
    // 
    // StoreInterleaved2(v_idx, v_rk) 会产生:
    // [v_idx[0], v_rk[0], v_idx[1], v_rk[1] ...]
    // 这正好对应 uint64_t 数组中的:
    // packed[0] = (rk[0]<<32)|idx[0]
    // packed[1] = (rk[1]<<32)|idx[1]
    // 完全省去了 Shift 和 Or 指令！
    
    hn::StoreInterleaved2(v_idx, v_rk, du, reinterpret_cast<U32*>(packed + i));
  }
#endif

  // Scalar fallback
  for (; i < n; ++i) {
    U32 rk = to_radix_key_scalar<K>(in[i].first);
    if (!ascending) rk = ~rk;
    packed[i] = (uint64_t(rk) << 32) | uint64_t(static_cast<U32>(i));
  }
}

template <typename K, typename V>
static inline void pack_k64_idx_simd(const parlay::KVPair<K, V>* in,
                                     size_t n,
                                     hwy::uint128_t* packed,
                                     bool ascending) {
  using Pair = parlay::KVPair<K, V>;
  static_assert(sizeof(Pair) == 16, "K64V64 requires 16-byte pair (tight layout)");

  using U64 = uint64_t;
  const hn::ScalableTag<U64> du;
  const size_t lanes = hn::Lanes(du);

  size_t i = 0;

#if HWY_TARGET != HWY_SCALAR
  for (; i + lanes <= n; i += lanes) {
    // AoS memory as interleaved 64-bit words: [key0,val0,key1,val1,...]
    const U64* src = reinterpret_cast<const U64*>(in + i);

    auto v_key_bits = hn::Zero(du);
    auto v_dummy    = hn::Zero(du);
    hn::LoadInterleaved2(du, src, v_key_bits, v_dummy);

    auto v_rk = to_radix_key_vec_bits<K>(du, v_key_bits);
    if (!ascending) {
      const auto all_ones = hn::Set(du, static_cast<U64>(~U64(0)));
      v_rk = hn::Xor(v_rk, all_ones);
    }

    const auto v_idx = hn::Iota(du, U64(i));

    // interleaved store to uint128 layout: [idx0,key0, idx1,key1, ...]
    U64* dst64 = reinterpret_cast<U64*>(packed + i);
    hn::StoreInterleaved2(v_idx, v_rk, du, dst64);
  }
#endif

  for (; i < n; ++i) {
    U64 rk = to_radix_key_scalar<K>(in[i].first);
    if (!ascending) rk = ~rk;
    packed[i].lo = static_cast<U64>(i);
    packed[i].hi = rk;
  }
}
template <typename Pair>
static inline void materialize_k32_L2(const Pair* HWY_RESTRICT in, 
                                      size_t n, 
                                      uint64_t* HWY_RESTRICT packed) {
    // 静态检查
    static_assert(sizeof(Pair) == 8, "K32V32 expects 8-byte pair");
    
    // 把输入看作 uint64 数组，方便直接拷贝 8 字节
    const uint64_t* src_ptr = reinterpret_cast<const uint64_t*>(in);
    
    size_t i = 0;

    // ==========================================================
    // 核心策略：8路标量展开 (Scalar Unroll 8x)
    // 
    // 编译器会将其翻译为：
    //   mov    r1, [packed + 0]   ; 读 (Key|Index)
    //   mov    r2, [packed + 8]
    //   ...
    //   mov    eax, r1d           ; 隐式截断取低32位 Index
    //   mov    ebx, r2d
    //   ...
    //   mov    rax, [src_ptr + rax*8] ; 读 L2 数据 (Index Load)
    //   mov    rbx, [src_ptr + rbx*8]
    //   ...
    //   mov    [packed + 0], rax  ; 写回
    //   mov    [packed + 8], rbx
    // 
    // 在 L2 命中场景下，这种指令流的吞吐量远高于 vpgatherqq
    // ==========================================================
    for (; i + 8 <= n; i += 8) {
        // 1. 提取 Index (利用 cast 截断，汇编里就是取低32位寄存器，零开销)
        uint32_t idx0 = static_cast<uint32_t>(packed[i + 0]);
        uint32_t idx1 = static_cast<uint32_t>(packed[i + 1]);
        uint32_t idx2 = static_cast<uint32_t>(packed[i + 2]);
        uint32_t idx3 = static_cast<uint32_t>(packed[i + 3]);
        uint32_t idx4 = static_cast<uint32_t>(packed[i + 4]);
        uint32_t idx5 = static_cast<uint32_t>(packed[i + 5]);
        uint32_t idx6 = static_cast<uint32_t>(packed[i + 6]);
        uint32_t idx7 = static_cast<uint32_t>(packed[i + 7]);

        // 2. 随机读取 (L2 Latency ~14 cycles)
        uint64_t v0 = src_ptr[idx0];
        uint64_t v1 = src_ptr[idx1];
        uint64_t v2 = src_ptr[idx2];
        uint64_t v3 = src_ptr[idx3];
        uint64_t v4 = src_ptr[idx4];
        uint64_t v5 = src_ptr[idx5];
        uint64_t v6 = src_ptr[idx6];
        uint64_t v7 = src_ptr[idx7];

        // 3. 写回
        packed[i + 0] = v0;
        packed[i + 1] = v1;
        packed[i + 2] = v2;
        packed[i + 3] = v3;
        packed[i + 4] = v4;
        packed[i + 5] = v5;
        packed[i + 6] = v6;
        packed[i + 7] = v7;
    }

    // 处理剩余元素
    for (; i < n; ++i) {
        uint32_t idx = static_cast<uint32_t>(packed[i]);
        packed[i] = src_ptr[idx];
    }
}
template <typename Pair>
static inline void materialize_k32(const Pair* HWY_RESTRICT in, 
                                   size_t n, 
                                   uint64_t* HWY_RESTRICT packed) {
    static_assert(sizeof(Pair) == 8, "K32V32 expects 8-byte pair");
    size_t i = 0;

    // =========================================================
    // 3. Scalar Fallback (Prefetch + Unroll)
    //    处理尾部数据，或不支持 AVX2 的机器
    // =========================================================
    constexpr size_t PREFETCH_DIST = 16; // 标量循环可以用小一点的预取

    for (; i + 4 <= n; i += 4) {
        if (i + 4 + PREFETCH_DIST < n) {
             uint32_t next_idx0 = static_cast<uint32_t>(packed[i + PREFETCH_DIST]);
             uint32_t next_idx2 = static_cast<uint32_t>(packed[i + PREFETCH_DIST + 2]);
             __builtin_prefetch(in + next_idx0, 0, 3);
             __builtin_prefetch(in + next_idx2, 0, 3);
        }

        // 保持你原有的 Unroll 写法
        const uint32_t s0 = static_cast<uint32_t>(packed[i + 0]);
        const uint32_t s1 = static_cast<uint32_t>(packed[i + 1]);
        const uint32_t s2 = static_cast<uint32_t>(packed[i + 2]);
        const uint32_t s3 = static_cast<uint32_t>(packed[i + 3]);

        const Pair p0 = in[s0];
        const Pair p1 = in[s1];
        const Pair p2 = in[s2];
        const Pair p3 = in[s3];

        std::memcpy(&packed[i + 0], &p0, sizeof(Pair));
        std::memcpy(&packed[i + 1], &p1, sizeof(Pair));
        std::memcpy(&packed[i + 2], &p2, sizeof(Pair));
        std::memcpy(&packed[i + 3], &p3, sizeof(Pair));
    }

    for (; i < n; ++i) {
        const uint32_t s = static_cast<uint32_t>(packed[i]);
        const Pair p = in[s];
        std::memcpy(&packed[i], &p, sizeof(Pair));
    }
}
// 针对 L2 Cache 优化的 Materialize K64
template <typename Pair>
static inline void materialize_k64_L2(const Pair* HWY_RESTRICT in, 
                                      size_t n, 
                                      hwy::uint128_t* HWY_RESTRICT packed) {
    // 强制把 packed 当作 Pair 数组处理
    // 这样编译器生成的代码就是 load(16B) -> store(16B)
    auto* dst = reinterpret_cast<Pair*>(packed);
    
    size_t i = 0;

    // 8路展开：让 CPU 的 OoO 引擎一次性看到 8 个独立的 Load 请求
    // 在 L2 命中场景下，这能把 Load 端口打满
    for (; i + 8 <= n; i += 8) {
        // 读取索引 (从 packed 的低 64 位)
        size_t s0 = static_cast<size_t>(packed[i+0].lo);
        size_t s1 = static_cast<size_t>(packed[i+1].lo);
        size_t s2 = static_cast<size_t>(packed[i+2].lo);
        size_t s3 = static_cast<size_t>(packed[i+3].lo);
        size_t s4 = static_cast<size_t>(packed[i+4].lo);
        size_t s5 = static_cast<size_t>(packed[i+5].lo);
        size_t s6 = static_cast<size_t>(packed[i+6].lo);
        size_t s7 = static_cast<size_t>(packed[i+7].lo);

        // 直接结构体赋值，最纯粹的 movups
        dst[i+0] = in[s0];
        dst[i+1] = in[s1];
        dst[i+2] = in[s2];
        dst[i+3] = in[s3];
        dst[i+4] = in[s4];
        dst[i+5] = in[s5];
        dst[i+6] = in[s6];
        dst[i+7] = in[s7];
    }

    for (; i < n; ++i) {
        size_t s = static_cast<size_t>(packed[i].lo);
        dst[i] = in[s];
    }
}
// --- K64V64 Implementation ---
template <typename Pair>
static inline void materialize_k64(const Pair* HWY_RESTRICT in, 
                                   size_t n, 
                                   hwy::uint128_t* HWY_RESTRICT packed) {
    static_assert(sizeof(Pair) == 16, "K64V64 expects 16-byte pair");
    size_t i = 0;

    // 对于 16字节对象，AVX512 没有原生的 "Gather 128-bit"。
    // 模拟 Gather 需要两次 64-bit Gather，指令数较多，收益不如标量+预取明显。
    // 因此这里主要依靠 Unroll + Software Prefetching。

    constexpr size_t PREFETCH_DIST = 12; // 128-bit 数据较大，预取距离稍短一点
    
    // 4-way unroll loop
    for (; i + 4 <= n; i += 4) {
        // --- Prefetching ---
        if (i + 4 + PREFETCH_DIST < n) {
             // 仅读取 Index 部分 (lo)
             size_t next_s0 = static_cast<size_t>(packed[i + PREFETCH_DIST + 0].lo);
             size_t next_s1 = static_cast<size_t>(packed[i + PREFETCH_DIST + 1].lo);
             // 预取 16 字节 Pair (通常在一个 Cache Line 内，但也可能跨线)
             __builtin_prefetch(in + next_s0, 0, 3);
             __builtin_prefetch(in + next_s1, 0, 3);
        }

        // --- Scalar Load/Store ---
        const size_t s0 = static_cast<size_t>(packed[i + 0].lo);
        const size_t s1 = static_cast<size_t>(packed[i + 1].lo);
        const size_t s2 = static_cast<size_t>(packed[i + 2].lo);
        const size_t s3 = static_cast<size_t>(packed[i + 3].lo);

        // 编译器通常会将这里的 memcpy 优化为 vmovdqu (AVX)
        std::memcpy(&packed[i + 0], in + s0, 16);
        std::memcpy(&packed[i + 1], in + s1, 16);
        std::memcpy(&packed[i + 2], in + s2, 16);
        std::memcpy(&packed[i + 3], in + s3, 16);
    }

    // Tail
    for (; i < n; ++i) {
        const size_t s = static_cast<size_t>(packed[i].lo);
        std::memcpy(&packed[i], in + s, 16);
    }
}

template <typename Pair>
static inline void materialize_from_index_to_same_buffer(
    const Pair* in, size_t n, uint64_t* packed) {
  static_assert(sizeof(Pair) == 8, "K32V32 path expects 8-byte pair");

  size_t i = 0;

  // 4-way unroll
  for (; i + 4 <= n; i += 4) {
    const uint32_t s0 = static_cast<uint32_t>(packed[i + 0]);
    const uint32_t s1 = static_cast<uint32_t>(packed[i + 1]);
    const uint32_t s2 = static_cast<uint32_t>(packed[i + 2]);
    const uint32_t s3 = static_cast<uint32_t>(packed[i + 3]);

    const Pair p0 = in[s0];
    const Pair p1 = in[s1];
    const Pair p2 = in[s2];
    const Pair p3 = in[s3];

    // 用 memcpy，避免 strict-aliasing 问题
    std::memcpy(&packed[i + 0], &p0, sizeof(Pair));
    std::memcpy(&packed[i + 1], &p1, sizeof(Pair));
    std::memcpy(&packed[i + 2], &p2, sizeof(Pair));
    std::memcpy(&packed[i + 3], &p3, sizeof(Pair));
  }

  for (; i < n; ++i) {
    const uint32_t s = static_cast<uint32_t>(packed[i]);
    const Pair p = in[s];
    std::memcpy(&packed[i], &p, sizeof(Pair));
  }
}

template <typename Pair>
static inline void materialize_from_index_to_same_buffer(
    const Pair* HWY_RESTRICT in, size_t n, hwy::uint128_t* HWY_RESTRICT packed) {
  static_assert(sizeof(Pair) == 16, "K64V64 path expects 16-byte pair");

  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const size_t s0 = static_cast<size_t>(packed[i + 0].lo);
    const size_t s1 = static_cast<size_t>(packed[i + 1].lo);
    const size_t s2 = static_cast<size_t>(packed[i + 2].lo);
    const size_t s3 = static_cast<size_t>(packed[i + 3].lo);

    std::memcpy(&packed[i + 0], in + s0, sizeof(Pair));
    std::memcpy(&packed[i + 1], in + s1, sizeof(Pair));
    std::memcpy(&packed[i + 2], in + s2, sizeof(Pair));
    std::memcpy(&packed[i + 3], in + s3, sizeof(Pair));
  }

  for (; i < n; ++i) {
    const size_t s = static_cast<size_t>(packed[i].lo);
    std::memcpy(&packed[i], in + s, sizeof(Pair));
  }
}

// =========================================================
// Stable sort AoS pair<K,V> using external buffer
// Buffer requirement:
//   K32V32 -> n * sizeof(uint64_t)      (== n * 8)
//   K64V64 -> n * sizeof(hwy::uint128_t)(== n *16)
// =========================================================
// =========================================================
// Stable sort AoS pair<K,V> using external buffer
// Template Arg:
//   CopyBack: true  -> 结果拷贝回 in (默认行为)
//             false -> 结果保留在 buffer 中 (in 保持原样或变为乱序)
// =========================================================
template <bool CopyBack = true, typename PairType>
void stable_sort_pairs_hwy(PairType* in,
                           size_t n,
                           void* buffer,
                           bool ascending = true) {
    // -----------------------------------------------------
    // 类型检查与提取
    // -----------------------------------------------------
    // 1. 强制要求是 KVPair
    static_assert(is_kvpair_v<PairType>, "Input must be parlay::KVPair<K, V>");

    // 2. 提取 K 和 V
    using K = typename GetKVType<PairType>::K;
    using V = typename GetKVType<PairType>::V;

    // 3. K/V 属性检查
    static_assert(std::is_arithmetic_v<K>, "K must be arithmetic");
    static_assert(std::is_arithmetic_v<V>, "V must be arithmetic");
    static_assert(sizeof(K) == sizeof(V), "Require sizeof(K) == sizeof(V)");
    static_assert(sizeof(K) == 4 || sizeof(K) == 8, "Only K32V32 or K64V64 supported");
    
    // 4. 内存布局检查
    // 确保 KVPair 是标准布局且无 Padding，保证 SIMD 加载安全
    static_assert(std::is_standard_layout_v<PairType>, "KVPair must be standard-layout");
    static_assert(sizeof(PairType) == 2 * sizeof(K), "KVPair must be tightly packed");
    static_assert(std::is_trivially_copyable_v<PairType>, "KVPair must be trivially copyable");

    // -----------------------------------------------------
    // 逻辑实现
    // -----------------------------------------------------
    if (n <= 1) {
        // 如果不拷回原数组，且 n=1，必须手动把这一个元素拷贝到 buffer
        // 这样调用者去 buffer 读数据时才不会读到垃圾值
        if constexpr (!CopyBack) {
            if (n == 1) {
                 std::memcpy(buffer, in, sizeof(PairType)); 
            }
        }
        return;
    }
  
    assert(in != nullptr);
    assert(buffer != nullptr);
// K32V32 Path
    if constexpr (sizeof(K) == 4) {
         assert(n <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()));

        auto* packed = reinterpret_cast<uint64_t*>(buffer);
        assert((reinterpret_cast<uintptr_t>(packed) % alignof(uint64_t)) == 0);
        
        // 1. Pack (Optimized with StoreInterleaved)
        pack_k32_idx_simd<K, V>(in, n, packed, ascending);

        // 2. Sort
        hwy::VQSort(packed, n, hwy::SortAscending());

        // 3. Materialize (Optimized with AVX512 Gather / Prefetch)
        materialize_k32_L2(in, n, packed);

// 4. CopyBack (修复 Segfault 的关键点)
        // if constexpr (CopyBack) {
        //     const hn::ScalableTag<uint64_t> d;
        //     const size_t lanes = hn::Lanes(d);
        //     size_t j = 0;
            
        //     auto* src_u64 = reinterpret_cast<const uint64_t*>(packed);
        //     auto* dst_u64 = reinterpret_cast<uint64_t*>(in);
            
        //     size_t total_u64 = n * (sizeof(PairType) / 8);
            
        //     for (; j + 4 * lanes <= total_u64; j += 4 * lanes) {
        //         // [FIX] 全部改为 LoadU / StoreU
        //         auto v0 = hn::LoadU(d, src_u64 + j + 0 * lanes);
        //         auto v1 = hn::LoadU(d, src_u64 + j + 1 * lanes);
        //         auto v2 = hn::LoadU(d, src_u64 + j + 2 * lanes);
        //         auto v3 = hn::LoadU(d, src_u64 + j + 3 * lanes);
                
        //         hn::StoreU(v0, d, dst_u64 + j + 0 * lanes);
        //         hn::StoreU(v1, d, dst_u64 + j + 1 * lanes);
        //         hn::StoreU(v2, d, dst_u64 + j + 2 * lanes);
        //         hn::StoreU(v3, d, dst_u64 + j + 3 * lanes);
        //     }
        //     // 处理剩余
        //     for (; j < total_u64; ++j) {
        //         dst_u64[j] = src_u64[j];
        //     }
        // }
        if constexpr (CopyBack) {
    // std::memcpy 会自动处理内存对齐、指令选择 (AVX/SSE/RepMovsb)
    // 对于 Trivially Copyable 的类型，这是最快且最安全的
    std::memcpy(in, packed, n * sizeof(PairType));
}
    } 
    // K64V64 Path
    else {
        auto* packed = reinterpret_cast<hwy::uint128_t*>(buffer);
        assert((reinterpret_cast<uintptr_t>(packed) % alignof(hwy::uint128_t)) == 0);

        // 1. Pack
        pack_k64_idx_simd<K, V>(in, n, packed, ascending);

        // 2. Sort
        hwy::VQSort(packed, n, hwy::SortAscending());

        // 3. Materialize (Optimized with Prefetch)
        materialize_k64_L2(in, n, packed);

        // 4. CopyBack (修复 Segfault 的关键点)
        // if constexpr (CopyBack) {
        //     const hn::ScalableTag<uint64_t> d;
        //     const size_t lanes = hn::Lanes(d);
        //     size_t j = 0;
            
        //     auto* src_u64 = reinterpret_cast<const uint64_t*>(packed);
        //     auto* dst_u64 = reinterpret_cast<uint64_t*>(in);
            
        //     size_t total_u64 = n * (sizeof(PairType) / 8);
            
        //     for (; j + 4 * lanes <= total_u64; j += 4 * lanes) {
        //         // [FIX] 全部改为 LoadU / StoreU
        //         auto v0 = hn::LoadU(d, src_u64 + j + 0 * lanes);
        //         auto v1 = hn::LoadU(d, src_u64 + j + 1 * lanes);
        //         auto v2 = hn::LoadU(d, src_u64 + j + 2 * lanes);
        //         auto v3 = hn::LoadU(d, src_u64 + j + 3 * lanes);
                
        //         hn::StoreU(v0, d, dst_u64 + j + 0 * lanes);
        //         hn::StoreU(v1, d, dst_u64 + j + 1 * lanes);
        //         hn::StoreU(v2, d, dst_u64 + j + 2 * lanes);
        //         hn::StoreU(v3, d, dst_u64 + j + 3 * lanes);
        //     }
        //     // 处理剩余
        //     for (; j < total_u64; ++j) {
        //         dst_u64[j] = src_u64[j];
        //     }
        // }
        if constexpr (CopyBack) {
    // std::memcpy 会自动处理内存对齐、指令选择 (AVX/SSE/RepMovsb)
    // 对于 Trivially Copyable 的类型，这是最快且最安全的
    std::memcpy(in, packed, n * sizeof(PairType));
}
    }
    
}

// =========================================================
// 3. Slice Wrapper
// =========================================================
template <bool CopyBack = true, typename InIterator, typename BufIterator>
void stable_sort_pairs_hwy(parlay::slice<InIterator, InIterator> In,
                           parlay::slice<BufIterator, BufIterator> Buffer,
                           bool ascending = true) {
    // 1. 获取原始类型
    using PairRaw = std::remove_cv_t<typename std::iterator_traits<InIterator>::value_type>;
    using BufT    = std::remove_cv_t<typename std::iterator_traits<BufIterator>::value_type>;

    // 2. 核心检查：必须是 KVPair
    static_assert(is_kvpair_v<PairRaw>, "Slice elements must be parlay::KVPair<K,V>");

    // 3. 提取 K 类型用于计算 Buffer 大小
    using K = typename GetKVType<PairRaw>::K;
    
    const size_t n = In.size();
  
    // 边界情况处理
    if (n <= 1) {
        if constexpr (!CopyBack) {
            if (n == 1) {
                // 将单个元素拷贝到 buffer
                auto* buf_ptr = std::addressof(*Buffer.begin());
                std::memcpy(buf_ptr, std::addressof(*In.begin()), sizeof(PairRaw));
            }
        }
        return;
    }
  
    // 4. Buffer 大小检查
    // K32 -> 每个元素需要 8字节 (uint64)
    // K64 -> 每个元素需要 16字节 (uint128)
    const size_t need_bytes = (sizeof(K) == 4) ? (n * 8) : (n * 16);
    const size_t buf_bytes  = Buffer.size() * sizeof(BufT);
    
    assert(buf_bytes >= need_bytes && "Stable buffer too small");

    // 5. 获取 Buffer 指针并转换
    auto* buf_ptr0 = std::addressof(*Buffer.begin());
    auto* buf_mut  = const_cast<std::remove_const_t<BufT>*>(buf_ptr0);

    // 6. 对齐检查 (Highway 必须 16 字节对齐)
    // 注意：即使 K32 使用 uint64 存储，VQSort 内部往往也偏好 16 字节对齐
    assert((reinterpret_cast<uintptr_t>(buf_mut) % 16) == 0 &&
           "Stable buffer must be 16-byte aligned");

    // 7. 调用实现
    // 自动推导 PairType = parlay::KVPair<K,V>
    stable_sort_pairs_hwy<CopyBack>(
        std::addressof(*In.begin()), 
        n, 
        reinterpret_cast<void*>(buf_mut), 
        ascending
    );
}
// unstable begins here

// ---------------------------------------------------------
// Pass 1: [K, V] -> Transform K -> Swap -> [V, RadixK]
// ---------------------------------------------------------
template <typename K, typename V>
void transform_forward_and_swap_hwy(parlay::KVPair<K, V>* data, size_t n) {
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
void restore_backward_and_swap_hwy(parlay::KVPair<K, V>* data, size_t n) {
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
void unstable_sort_pairs_hwy(parlay::KVPair<K, V>* in,
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
    static_assert(is_kvpair_v<T>, "Must be parlay::KVPair");
    using K = typename GetKVType<T>::K;
    using V = typename GetKVType<T>::V;
    
    unstable_sort_pairs_hwy<K, V>(In.begin(), In.size(), ascending);
}
} // namespace internal
} // namespace parlay
