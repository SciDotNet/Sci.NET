//
// Created by reece on 02/08/2023.
//

#include "cuda_arithmetic.h"
#include <cuda_bf16.h>

sdnApiStatusCode add_tensor_tensor_bf16(nv_bfloat16 *a, nv_bfloat16 *b, nv_bfloat16 *result, int64_t n) {
    return add_tensor_tensor_bf16_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_tensor_fp32(float *a, float *b, float *result, int64_t n) {
    return add_tensor_tensor_fp32_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_tensor_fp64(double *a, double *b, double *result, int64_t n) {
    return add_tensor_tensor_fp64_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t n) {
    return add_tensor_tensor_u8_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t n) {
    return add_tensor_tensor_i8_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t n) {
    return add_tensor_tensor_u16_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t n) {
    return add_tensor_tensor_i16_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t n) {
    return add_tensor_tensor_u32_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t n) {
    return add_tensor_tensor_i32_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t n) {
    return add_tensor_tensor_u64_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t n) {
    return add_tensor_tensor_i64_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_broadcast_tensor_fp32(float *a, float *b, float *result, int64_t m, int64_t n) {
    return add_tensor_broadcast_tensor_fp32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_broadcast_tensor_fp64(double *a, double *b, double *result, int64_t m, int64_t n) {
    return add_tensor_broadcast_tensor_fp64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_broadcast_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n) {
    return add_tensor_broadcast_tensor_u8_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_broadcast_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n) {
    return add_tensor_broadcast_tensor_i8_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_broadcast_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n) {
    return add_tensor_broadcast_tensor_u16_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_broadcast_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n) {
    return add_tensor_broadcast_tensor_i16_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_broadcast_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n) {
    return add_tensor_broadcast_tensor_u32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_broadcast_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n) {
    return add_tensor_broadcast_tensor_i32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_broadcast_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n) {
    return add_tensor_broadcast_tensor_u64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_tensor_broadcast_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n) {
    return add_tensor_broadcast_tensor_i64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_broadcast_tensor_tensor_fp32(float *a, float *b, float *result, int64_t m, int64_t n) {
    return add_broadcast_tensor_tensor_fp32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_broadcast_tensor_tensor_fp64(double *a, double *b, double *result, int64_t m, int64_t n) {
    return add_broadcast_tensor_tensor_fp64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_broadcast_tensor_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n) {
    return add_broadcast_tensor_tensor_u8_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_broadcast_tensor_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n) {
    return add_broadcast_tensor_tensor_i8_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_broadcast_tensor_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n) {
    return add_broadcast_tensor_tensor_u16_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_broadcast_tensor_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n) {
    return add_broadcast_tensor_tensor_i16_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_broadcast_tensor_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n) {
    return add_broadcast_tensor_tensor_u32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_broadcast_tensor_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n) {
    return add_broadcast_tensor_tensor_i32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_broadcast_tensor_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n) {
    return add_broadcast_tensor_tensor_u64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode add_broadcast_tensor_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n) {
    return add_broadcast_tensor_tensor_i64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_tensor_fp32(float *a, float *b, float *result, int64_t n) {
    return subtract_tensor_tensor_fp32_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_tensor_fp64(double *a, double *b, double *result, int64_t n) {
    return subtract_tensor_tensor_fp64_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t n) {
    return subtract_tensor_tensor_u8_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t n) {
    return subtract_tensor_tensor_i8_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t n) {
    return subtract_tensor_tensor_u16_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t n) {
    return subtract_tensor_tensor_i16_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t n) {
    return subtract_tensor_tensor_u32_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t n) {
    return subtract_tensor_tensor_i32_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t n) {
    return subtract_tensor_tensor_u64_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t n) {
    return subtract_tensor_tensor_i64_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_broadcast_tensor_fp32(float *a, float *b, float *result, int64_t m, int64_t n) {
    return subtract_tensor_broadcast_tensor_fp32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_broadcast_tensor_fp64(double *a, double *b, double *result, int64_t m, int64_t n) {
    return subtract_tensor_broadcast_tensor_fp64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_broadcast_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n) {
    return subtract_tensor_broadcast_tensor_u8_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_broadcast_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n) {
    return subtract_tensor_broadcast_tensor_i8_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_broadcast_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n) {
    return subtract_tensor_broadcast_tensor_u16_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_broadcast_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n) {
    return subtract_tensor_broadcast_tensor_i16_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_broadcast_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n) {
    return subtract_tensor_broadcast_tensor_u32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_broadcast_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n) {
    return subtract_tensor_broadcast_tensor_i32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_broadcast_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n) {
    return subtract_tensor_broadcast_tensor_u64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_tensor_broadcast_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n) {
    return subtract_tensor_broadcast_tensor_i64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_broadcast_tensor_tensor_fp32(float *a, float *b, float *result, int64_t m, int64_t n) {
    return subtract_broadcast_tensor_tensor_fp32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_broadcast_tensor_tensor_fp64(double *a, double *b, double *result, int64_t m, int64_t n) {
    return subtract_broadcast_tensor_tensor_fp64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_broadcast_tensor_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n) {
    return subtract_broadcast_tensor_tensor_u8_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_broadcast_tensor_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n) {
    return subtract_broadcast_tensor_tensor_i8_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_broadcast_tensor_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n) {
    return subtract_broadcast_tensor_tensor_u16_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_broadcast_tensor_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n) {
    return subtract_broadcast_tensor_tensor_i16_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_broadcast_tensor_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n) {
    return subtract_broadcast_tensor_tensor_u32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_broadcast_tensor_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n) {
    return subtract_broadcast_tensor_tensor_i32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_broadcast_tensor_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n) {
    return subtract_broadcast_tensor_tensor_u64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode subtract_broadcast_tensor_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n) {
    return subtract_broadcast_tensor_tensor_i64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_tensor_fp32(float *a, float *b, float *result, int64_t n) {
    return multiply_tensor_tensor_fp32_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_tensor_fp64(double *a, double *b, double *result, int64_t n) {
    return multiply_tensor_tensor_fp64_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t n) {
    return multiply_tensor_tensor_u8_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t n) {
    return multiply_tensor_tensor_i8_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t n) {
    return multiply_tensor_tensor_u16_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t n) {
    return multiply_tensor_tensor_i16_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t n) {
    return multiply_tensor_tensor_u32_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t n) {
    return multiply_tensor_tensor_i32_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t n) {
    return multiply_tensor_tensor_u64_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t n) {
    return multiply_tensor_tensor_i64_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_broadcast_tensor_fp32(float *a, float *b, float *result, int64_t m, int64_t n) {
    return multiply_tensor_broadcast_tensor_fp32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_broadcast_tensor_fp64(double *a, double *b, double *result, int64_t m, int64_t n) {
    return multiply_tensor_broadcast_tensor_fp64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_broadcast_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n) {
    return multiply_tensor_broadcast_tensor_u8_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_broadcast_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n) {
    return multiply_tensor_broadcast_tensor_i8_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_broadcast_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n) {
    return multiply_tensor_broadcast_tensor_u16_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_broadcast_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n) {
    return multiply_tensor_broadcast_tensor_i16_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_broadcast_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n) {
    return multiply_tensor_broadcast_tensor_u32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_broadcast_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n) {
    return multiply_tensor_broadcast_tensor_i32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_broadcast_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n) {
    return multiply_tensor_broadcast_tensor_u64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_tensor_broadcast_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n) {
    return multiply_tensor_broadcast_tensor_i64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_broadcast_tensor_tensor_fp32(float *a, float *b, float *result, int64_t m, int64_t n) {
    return multiply_broadcast_tensor_tensor_fp32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_broadcast_tensor_tensor_fp64(double *a, double *b, double *result, int64_t m, int64_t n) {
    return multiply_broadcast_tensor_tensor_fp64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_broadcast_tensor_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n) {
    return multiply_broadcast_tensor_tensor_u8_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_broadcast_tensor_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n) {
    return multiply_broadcast_tensor_tensor_i8_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_broadcast_tensor_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n) {
    return multiply_broadcast_tensor_tensor_u16_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_broadcast_tensor_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n) {
    return multiply_broadcast_tensor_tensor_i16_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_broadcast_tensor_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n) {
    return multiply_broadcast_tensor_tensor_u32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_broadcast_tensor_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n) {
    return multiply_broadcast_tensor_tensor_i32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_broadcast_tensor_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n) {
    return multiply_broadcast_tensor_tensor_u64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode multiply_broadcast_tensor_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n) {
    return multiply_broadcast_tensor_tensor_i64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_tensor_fp32(float *a, float *b, float *result, int64_t n) {
    return divide_tensor_tensor_fp32_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_tensor_fp64(double *a, double *b, double *result, int64_t n) {
    return divide_tensor_tensor_fp64_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t n) {
    return divide_tensor_tensor_u8_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t n) {
    return divide_tensor_tensor_i8_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t n) {
    return divide_tensor_tensor_u16_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t n) {
    return divide_tensor_tensor_i16_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t n) {
    return divide_tensor_tensor_u32_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t n) {
    return divide_tensor_tensor_i32_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t n) {
    return divide_tensor_tensor_u64_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t n) {
    return divide_tensor_tensor_i64_invoke(a, b, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_broadcast_tensor_fp32(float *a, float *b, float *result, int64_t m, int64_t n) {
    return divide_tensor_broadcast_tensor_fp32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_broadcast_tensor_fp64(double *a, double *b, double *result, int64_t m, int64_t n) {
    return divide_tensor_broadcast_tensor_fp64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_broadcast_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n) {
    return divide_tensor_broadcast_tensor_u8_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_broadcast_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n) {
    return divide_tensor_broadcast_tensor_i8_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_broadcast_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n) {
    return divide_tensor_broadcast_tensor_u16_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_broadcast_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n) {
    return divide_tensor_broadcast_tensor_i16_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_broadcast_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n) {
    return divide_tensor_broadcast_tensor_u32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_broadcast_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n) {
    return divide_tensor_broadcast_tensor_i32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_broadcast_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n) {
    return divide_tensor_broadcast_tensor_u64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_tensor_broadcast_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n) {
    return divide_tensor_broadcast_tensor_i64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_broadcast_tensor_tensor_fp32(float *a, float *b, float *result, int64_t m, int64_t n) {
    return divide_broadcast_tensor_tensor_fp32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_broadcast_tensor_tensor_fp64(double *a, double *b, double *result, int64_t m, int64_t n) {
    return divide_broadcast_tensor_tensor_fp64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_broadcast_tensor_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n) {
    return divide_broadcast_tensor_tensor_u8_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_broadcast_tensor_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n) {
    return divide_broadcast_tensor_tensor_i8_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_broadcast_tensor_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n) {
    return divide_broadcast_tensor_tensor_u16_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_broadcast_tensor_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n) {
    return divide_broadcast_tensor_tensor_i16_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_broadcast_tensor_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n) {
    return divide_broadcast_tensor_tensor_u32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_broadcast_tensor_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n) {
    return divide_broadcast_tensor_tensor_i32_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_broadcast_tensor_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n) {
    return divide_broadcast_tensor_tensor_u64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode divide_broadcast_tensor_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n) {
    return divide_broadcast_tensor_tensor_i64_invoke(a, b, result, m, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode negate_tensor_fp32(float *a, float *result, int64_t n) {
    return negate_tensor_fp32_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode negate_tensor_fp64(double *a, double *result, int64_t n) {
    return negate_tensor_fp64_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode negate_tensor_i8(int8_t *a, int8_t *result, int64_t n) {
    return negate_tensor_i8_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode negate_tensor_i16(int16_t *a, int16_t *result, int64_t n) {
    return negate_tensor_i16_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode negate_tensor_i32(int32_t *a, int32_t *result, int64_t n) {
    return negate_tensor_i32_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode negate_tensor_i64(int64_t *a, int64_t *result, int64_t n) {
    return negate_tensor_i64_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode abs_tensor_fp32(float *a, float *result, int64_t n) {
    return abs_tensor_fp32_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode abs_tensor_fp64(double *a, double *result, int64_t n) {
    return abs_tensor_fp64_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode abs_tensor_i8(int8_t *a, int8_t *result, int64_t n) {
    return abs_tensor_i8_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode abs_tensor_i16(int16_t *a, int16_t *result, int64_t n) {
    return abs_tensor_i16_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode abs_tensor_i32(int32_t *a, int32_t *result, int64_t n) {
    return abs_tensor_i32_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode abs_tensor_i64(int64_t *a, int64_t *result, int64_t n) {
    return abs_tensor_i64_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode sqrt_tensor_fp32(float *a, float *result, int64_t n) {
    return sqrt_tensor_fp32_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode sqrt_tensor_fp64(double *a, double *result, int64_t n) {
    return sqrt_tensor_fp64_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode sqrt_tensor_u8(uint8_t *a, uint8_t *result, int64_t n) {
    return sqrt_tensor_u8_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode sqrt_tensor_i8(int8_t *a, int8_t *result, int64_t n) {
    return sqrt_tensor_i8_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode sqrt_tensor_u16(uint16_t *a, uint16_t *result, int64_t n) {
    return sqrt_tensor_u16_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode sqrt_tensor_i16(int16_t *a, int16_t *result, int64_t n) {
    return sqrt_tensor_i16_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode sqrt_tensor_u32(uint32_t *a, uint32_t *result, int64_t n) {
    return sqrt_tensor_u32_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode sqrt_tensor_i32(int32_t *a, int32_t *result, int64_t n) {
    return sqrt_tensor_i32_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode sqrt_tensor_u64(uint64_t *a, uint64_t *result, int64_t n) {
    return sqrt_tensor_u64_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}

sdnApiStatusCode sqrt_tensor_i64(int64_t *a, int64_t *result, int64_t n) {
    return sqrt_tensor_i64_invoke(a, result, n) ? sdnSuccess : sdnInternalError;
}
