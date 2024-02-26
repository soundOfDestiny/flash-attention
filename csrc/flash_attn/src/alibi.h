#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_causal, typename Engine, typename Layout>
inline __device__ void apply_alibi(Tensor<Engine, Layout> &tensor, 
                                   const int col_idx_offset_,
                                   const int max_seqlen_k, 
                                   const int row_idx_offset,
                                   const int max_seqlen_q, 
                                   const int warp_row_stride,
                                   const float alibi_slope,
                                   const float alibi_exp) {
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    if (Is_causal && alibi_exp == 1.0) {  // Simpler, we add the same bias vector to all rows
        #pragma unroll
        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
            const int col_idx_base = col_idx_offset + nj * 8;
            #pragma unroll
            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                const int col_idx = col_idx_base + j;
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                }
            }
        }
    } else {  // Bias depends on both row_idx and col_idx
        #pragma unroll
        for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
            const int row_idx_base = row_idx_offset + mi * warp_row_stride;
            #pragma unroll
            for (int i = 0; i < size<0, 0>(tensor); ++i) {
                const int row_idx = row_idx_base + i * 8;
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        tensor(make_coord(i, mi), make_coord(j, nj)) -= alibi_slope * powf(abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx), alibi_exp);
                    }
                }
            }
        }
    }
}

template <bool Is_causal, typename Engine, typename Layout>
inline __device__ void apply_alibi_slope_grad(float &alibi_slope_grad,
                                              Tensor<Engine, Layout> &tensor,
                                              const int col_idx_offset_,
                                              const int max_seqlen_k,
                                              const int row_idx_offset,
                                              const int max_seqlen_q,
                                              const int warp_row_stride,
                                              const float alibi_slope,
                                              const float alibi_exp) {
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int row_idx = row_idx_base + i * 8;
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    const int dis = abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                    alibi_slope_grad -= tensor(make_coord(i, mi), make_coord(j, nj)) * powf(dis, alibi_exp);
                }
            }
        }
    }
}

template <bool Is_causal, typename Engine, typename Layout>
inline __device__ void apply_alibi_exp_grad(float &alibi_exp_grad,
                                            Tensor<Engine, Layout> &tensor,
                                            const int col_idx_offset_,
                                            const int max_seqlen_k,
                                            const int row_idx_offset,
                                            const int max_seqlen_q,
                                            const int warp_row_stride,
                                            const float alibi_slope,
                                            const float alibi_exp) {
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int row_idx = row_idx_base + i * 8;
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    const int dis = abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                    const float log_dis = dis ? logf(dis) : 0;
                    alibi_exp_grad -= tensor(make_coord(i, mi), make_coord(j, nj)) * alibi_slope * powf(dis, alibi_exp) * log_dis;
                }
            }
        }
    }
}

template <bool Is_causal, typename Engine, typename Layout>
inline __device__ void apply_alibi_var_slope(
    Tensor<Engine, Layout> &tensor, 
    const int col_idx_offset_,
    const int max_seqlen_k, 
    const int row_idx_offset,
    const int max_seqlen_q, 
    const int warp_row_stride,
    const float alibi_slope[][2],
    const float alibi_exp[][2],
    const bool seqlenq_ngroups_swapped
) {
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    if (seqlenq_ngroups_swapped) {
        #pragma unroll
        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
            const int col_idx_base = col_idx_offset + nj * 8;
            #pragma unroll
            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                const int col_idx = col_idx_base + j;
                #pragma unroll
                for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                    #pragma unroll
                    for (int i = 0; i < size<0, 0>(tensor); ++i) {
                        tensor(make_coord(i, mi), make_coord(j, nj)) -= alibi_slope[mi][i] * powf(max_seqlen_k - 1 - col_idx, alibi_exp[mi][i]);
                    }
                }
            }
        }
    } else {  // Bias depends on both row_idx and col_idx
        #pragma unroll
        for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
            const int row_idx_base = row_idx_offset + mi * warp_row_stride;
            #pragma unroll
            for (int i = 0; i < size<0, 0>(tensor); ++i) {
                const int row_idx = row_idx_base + i * 8;
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        tensor(make_coord(i, mi), make_coord(j, nj)) -= alibi_slope[mi][i] * powf(abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx), alibi_exp[mi][i]);
                    }
                }
            }
        }
    }
}

}  // namespace flash
