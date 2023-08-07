//
// Created by reece on 01/08/2023.
//

#ifndef SCI_NET_NATIVE_MATRIX_MULTIPLY_H
#define SCI_NET_NATIVE_MATRIX_MULTIPLY_H

#include "api.h"

SDN_DLL_EXPORT_API matrix_multiply_fp32(float *left,
                                        float *right,
                                        float *result,
                                        int32_t left_rows,
                                        int32_t left_cols,
                                        int32_t right_cols);

SDN_DLL_EXPORT_API matrix_multiply_fp64(double *left,
                                        double *right,
                                        double *result,
                                        int32_t left_rows,
                                        int32_t left_cols,
                                        int32_t right_cols);

SDN_DLL_EXPORT_API matrix_multiply_u8(uint8_t *left,
                                      uint8_t *right,
                                      uint8_t *result,
                                      int32_t left_rows,
                                      int32_t left_cols,
                                      int32_t right_cols);

SDN_DLL_EXPORT_API matrix_multiply_i8(int8_t *left,
                                      int8_t *right,
                                      int8_t *result,
                                      int32_t left_rows,
                                      int32_t left_cols,
                                      int32_t right_cols);

SDN_DLL_EXPORT_API matrix_multiply_u16(uint16_t *left,
                                       uint16_t *right,
                                       uint16_t *result,
                                       int32_t left_rows,
                                       int32_t left_cols,
                                       int32_t right_cols);

SDN_DLL_EXPORT_API matrix_multiply_i16(int16_t *left,
                                       int16_t *right,
                                       int16_t *result,
                                       int32_t left_rows,
                                       int32_t left_cols,
                                       int32_t right_cols);

SDN_DLL_EXPORT_API matrix_multiply_u32(uint32_t *left,
                                       uint32_t *right,
                                       uint32_t *result,
                                       int32_t left_rows,
                                       int32_t left_cols,
                                       int32_t right_cols);

SDN_DLL_EXPORT_API matrix_multiply_i32(int32_t *left,
                                       int32_t *right,
                                       int32_t *result,
                                       int32_t left_rows,
                                       int32_t left_cols,
                                       int32_t right_cols);

SDN_DLL_EXPORT_API matrix_multiply_u64(uint64_t *left,
                                       uint64_t *right,
                                       uint64_t *result,
                                       int32_t left_rows,
                                       int32_t left_cols,
                                       int32_t right_cols);

SDN_DLL_EXPORT_API matrix_multiply_i64(int64_t *left,
                                       int64_t *right,
                                       int64_t *result,
                                       int32_t left_rows,
                                       int32_t left_cols,
                                       int32_t right_cols);


#endif //SCI_NET_NATIVE_MATRIX_MULTIPLY_H
