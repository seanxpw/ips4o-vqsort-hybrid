#pragma once

#if defined(IPS4O_HYBRID_USE_PROJECT_BASE_CASE)
#include "../../../../src/modules/base_case/sort_hwy_pair.hpp"
#else
#include "sort_hwy_pair_local.hpp"
#endif

namespace parlay {
namespace internal_simd {

template <bool second_is_key = false, bool CopyBack = true, typename PairType>
void stable_sort_pairs_hwy(PairType* in,
                           size_t n,
                           void* buffer,
                           bool ascending = true) {
    ss_sort::detail::stable_sort_pairs_hwy<second_is_key, CopyBack>(
        in, n, buffer, ascending);
}

template <bool second_is_key = false,
          bool CopyBack = true,
          typename InIterator,
          typename BufIterator>
void stable_sort_pairs_hwy(parlay::slice<InIterator, InIterator> In,
                           parlay::slice<BufIterator, BufIterator> Buffer,
                           bool ascending = true) {
    ss_sort::detail::stable_sort_pairs_hwy<second_is_key, CopyBack>(
        In, Buffer, ascending);
}

template <bool second_is_key = false, typename K, typename V>
void unstable_sort_pairs_hwy(ss_sort::KVPair<K, V>* in,
                             size_t n,
                             bool ascending = true) {
    ss_sort::detail::unstable_sort_pairs_hwy<second_is_key, K, V>(
        in, n, ascending);
}

template <bool second_is_key = false, typename InIterator>
void unstable_sort_pairs_hwy(parlay::slice<InIterator, InIterator> In,
                             bool ascending = true) {
    ss_sort::detail::unstable_sort_pairs_hwy<second_is_key>(In, ascending);
}

}  // namespace internal_simd
}  // namespace parlay
