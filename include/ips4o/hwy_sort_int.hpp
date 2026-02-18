#pragma once
#include <hwy/highway.h>
#include <hwy/contrib/sort/vqsort.h>
#include <cstddef>
#include "parlay/slice.h"

namespace parlay {
namespace internal_simd {

// raw pointer 版本
template <typename T>
HWY_ATTR inline void unstable_sort_hwy_inplace(T* data, size_t n, bool ascending = true) {
  if (n <= 1) return;
  if (ascending) {
    hwy::VQSort(data, n, hwy::SortAscending());
  } else {
    hwy::VQSort(data, n, hwy::SortDescending());
  }
}

// parlay::slice 版本
template <typename Iterator>
HWY_ATTR inline void unstable_sort_hwy_inplace(slice<Iterator, Iterator> In,
                                               bool ascending = true) {
  using T = typename slice<Iterator, Iterator>::value_type;
  const size_t n = In.size();
  if (n <= 1) return;
  T* ptr = &(*In.begin());  // 假设 contiguous slice
  unstable_sort_hwy_inplace(ptr, n, ascending);
}

} // namespace internal
} // namespace parlay
