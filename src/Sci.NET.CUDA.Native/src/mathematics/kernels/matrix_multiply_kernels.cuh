//
// Created by reece on 01/08/2023.
//

#ifndef SCI_NET_NATIVE_MATRIX_MULTIPLY_KERNELS_H
#define SCI_NET_NATIVE_MATRIX_MULTIPLY_KERNELS_H

#include <cuda_bf16.h>

bool matrix_multiply_bf16_invoke(nv_bfloat16 *left,
                                 nv_bfloat16 *right,
                                 nv_bfloat16 *result,
                                 int32_t left_rows,
                                 int32_t left_cols,
                                 int32_t right_cols);

bool matrix_multiply_u8_invoke(uint8_t *left,
                               uint8_t *right,
                               uint8_t *result,
                               int32_t left_rows,
                               int32_t left_cols,
                               int32_t right_cols);

bool matrix_multiply_i8_invoke(int8_t *left,
                               int8_t *right,
                               int8_t *result,
                               int32_t left_rows,
                               int32_t left_cols,
                               int32_t right_cols);

bool matrix_multiply_u16_invoke(uint16_t *left,
                                uint16_t *right,
                                uint16_t *result,
                                int32_t left_rows,
                                int32_t left_cols,
                                int32_t right_cols);

bool matrix_multiply_i16_invoke(int16_t *left,
                                int16_t *right,
                                int16_t *result,
                                int32_t left_rows,
                                int32_t left_cols,
                                int32_t right_cols);

bool matrix_multiply_u32_invoke(uint32_t *left,
                                uint32_t *right,
                                uint32_t *result,
                                int32_t left_rows,
                                int32_t left_cols,
                                int32_t right_cols);

bool matrix_multiply_i32_invoke(int32_t *left,
                                int32_t *right,
                                int32_t *result,
                                int32_t left_rows,
                                int32_t left_cols,
                                int32_t right_cols);

bool matrix_multiply_u64_invoke(uint64_t *left,
                                uint64_t *right,
                                uint64_t *result,
                                int32_t left_rows,
                                int32_t left_cols,
                                int32_t right_cols);

bool matrix_multiply_i64_invoke(int64_t *left,
                                int64_t *right,
                                int64_t *result,
                                int32_t left_rows,
                                int32_t left_cols,
                                int32_t right_cols);

#endif //SCI_NET_NATIVE_MATRIX_MULTIPLY_KERNELS_H
