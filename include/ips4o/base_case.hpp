/******************************************************************************
 * include/ips4o/base_case.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 *
 ******************************************************************************
 * BSD 2-Clause License
 *
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include <type_traits>
#include <iterator>
#include <cstdint> // 包含 uint32_t, int64_t 等定义
#include "ips4o_fwd.hpp"
#include "utils.hpp"
#include "hwy_sort_int.hpp"
#include "sort_hwy_pair.hpp"
#include "parlay/slice.h"
namespace ips4o {
namespace detail {

/**
 * Insertion sort.
 */
template <class It, class Comp>
void insertionSort(const It begin, const It end, Comp comp) {
    IPS4OML_ASSUME_NOT(begin >= end);

    for (It it = begin + 1; it < end; ++it) {
        typename std::iterator_traits<It>::value_type val = std::move(*it);
        if (comp(val, *begin)) {
            std::move_backward(begin, it, it + 1);
            *begin = std::move(val);
        } else {
            auto cur = it;
            for (auto next = it - 1; comp(val, *next); --next) {
                *cur = std::move(*next);
                cur = next;
            }
            *cur = std::move(val);
        }
    }
}



// 1. 定义一个 Trait 来集中管理支持 SIMD 的类型
// 默认情况为 false
template <typename T>
struct is_simd_sortable : std::false_type {};

// 2. 特化白名单：将支持的类型设为 true
template <> struct is_simd_sortable<int32_t>  : std::true_type {};
template <> struct is_simd_sortable<uint32_t> : std::true_type {};
template <> struct is_simd_sortable<int64_t>  : std::true_type {};
template <> struct is_simd_sortable<uint64_t> : std::true_type {};
template <> struct is_simd_sortable<float>    : std::true_type {};
template <> struct is_simd_sortable<double>   : std::true_type {};

// 注意：如果你的平台 int 就是 int32_t，上面的特化会自动覆盖。
// 但为了防止某些平台定义不同，也可以显式加上原生类型：
// template <> struct is_simd_sortable<int> : std::true_type {};


/**
 * Wrapper for base case sorter.
 */
template <class It, class Comp>
inline void baseCaseSort(It begin, It end, Comp&& comp) {
    if (begin == end) return;

    using ValueType = typename std::iterator_traits<It>::value_type;

    // 3. 使用 if constexpr 进行编译期分发
    //is_simd_sortable<ValueType>::value 会在编译期计算出 true/false
    if constexpr (is_simd_sortable<ValueType>::value) {
        
        auto s = parlay::make_slice(begin, end);
        parlay::internal_simd::unstable_sort_hwy_inplace(s);
        
    } else {
        
        auto s = parlay::make_slice(begin, end);
        parlay::internal_simd::unstable_sort_pairs_hwy(s);
        
    }
}

template <class It, class Comp, class ThreadPool>
inline bool isSorted(It begin, It end, Comp&& comp, ThreadPool& thread_pool) {
    // Do nothing if input is already sorted.
    std::vector<bool> is_sorted(thread_pool.numThreads());
    thread_pool(
            [begin, end, &is_sorted, &comp](int my_id, int num_threads) {
                const auto size = end - begin;
                const auto stripe = (size + num_threads - 1) / num_threads;
                const auto my_begin = begin + std::min(stripe * my_id, size);
                const auto my_end = begin + std::min(stripe * (my_id + 1) + 1, size);
                is_sorted[my_id] = std::is_sorted(my_begin, my_end, comp);
            },
            thread_pool.numThreads());

    return std::all_of(is_sorted.begin(), is_sorted.end(), [](bool res) { return res; });
}

template <class It, class Comp>
inline bool sortSimpleCases(It begin, It end, Comp&& comp) {
    if (begin == end) {
        return true;
    }

    // If last element is not smaller than first element,
    // test if input is sorted (input is not reverse sorted).
    if (!comp(*(end - 1), *begin)) {
        if (std::is_sorted(begin, end, comp)) {
            return true;
        }
    } else {
        // Check whether the input is reverse sorted.
        for (It it = begin; (it + 1) != end; ++it) {
            if (comp(*it, *(it + 1))) {
                return false;
            }
        }
        std::reverse(begin, end);
        return true;
    }

    return false;
}

}  // namespace detail
}  // namespace ips4o
