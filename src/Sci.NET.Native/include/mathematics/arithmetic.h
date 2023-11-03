//
// Created by reece on 02/08/2023.
//

#ifndef SCI_NET_NATIVE_ARITHMETIC_H
#define SCI_NET_NATIVE_ARITHMETIC_H

#include "api.h"

SDN_DLL_EXPORT_API add_tensor_tensor_fp32(float *a, float *b, float *result, int64_t n);
SDN_DLL_EXPORT_API add_tensor_tensor_fp64(double *a, double *b, double *result, int64_t n);
SDN_DLL_EXPORT_API add_tensor_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t n);
SDN_DLL_EXPORT_API add_tensor_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t n);
SDN_DLL_EXPORT_API add_tensor_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t n);
SDN_DLL_EXPORT_API add_tensor_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t n);
SDN_DLL_EXPORT_API add_tensor_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t n);
SDN_DLL_EXPORT_API add_tensor_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t n);
SDN_DLL_EXPORT_API add_tensor_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t n);
SDN_DLL_EXPORT_API add_tensor_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t n);
SDN_DLL_EXPORT_API add_tensor_broadcast_tensor_fp32(float *a, float *b, float *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_tensor_broadcast_tensor_fp64(double *a, double *b, double *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_tensor_broadcast_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_tensor_broadcast_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_tensor_broadcast_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_tensor_broadcast_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_tensor_broadcast_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_tensor_broadcast_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_tensor_broadcast_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_tensor_broadcast_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_broadcast_tensor_tensor_fp32(float *a, float *b, float *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_broadcast_tensor_tensor_fp64(double *a, double *b, double *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_broadcast_tensor_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_broadcast_tensor_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_broadcast_tensor_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_broadcast_tensor_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_broadcast_tensor_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_broadcast_tensor_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_broadcast_tensor_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API add_broadcast_tensor_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n);

SDN_DLL_EXPORT_API subtract_tensor_tensor_fp32(float *a, float *b, float *result, int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_tensor_fp64(double *a, double *b, double *result, int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_broadcast_tensor_fp32(float *a, float *b, float *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_broadcast_tensor_fp64(double *a, double *b, double *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_broadcast_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_broadcast_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_broadcast_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m,
                                                        int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_broadcast_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_broadcast_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m,
                                                        int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_broadcast_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_broadcast_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m,
                                                        int64_t n);
SDN_DLL_EXPORT_API subtract_tensor_broadcast_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API subtract_broadcast_tensor_tensor_fp32(float *a, float *b, float *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API subtract_broadcast_tensor_tensor_fp64(double *a, double *b, double *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API subtract_broadcast_tensor_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API subtract_broadcast_tensor_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API subtract_broadcast_tensor_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m,
                                                        int64_t n);
SDN_DLL_EXPORT_API subtract_broadcast_tensor_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API subtract_broadcast_tensor_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m,
                                                        int64_t n);
SDN_DLL_EXPORT_API subtract_broadcast_tensor_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API subtract_broadcast_tensor_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m,
                                                        int64_t n);
SDN_DLL_EXPORT_API subtract_broadcast_tensor_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n);

SDN_DLL_EXPORT_API multiply_tensor_tensor_fp32(float *a, float *b, float *result, int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_tensor_fp64(double *a, double *b, double *result, int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_broadcast_tensor_fp32(float *a, float *b, float *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_broadcast_tensor_fp64(double *a, double *b, double *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_broadcast_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_broadcast_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_broadcast_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m,
                                                        int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_broadcast_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_broadcast_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m,
                                                        int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_broadcast_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_broadcast_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m,
                                                        int64_t n);
SDN_DLL_EXPORT_API multiply_tensor_broadcast_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API multiply_broadcast_tensor_tensor_fp32(float *a, float *b, float *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API multiply_broadcast_tensor_tensor_fp64(double *a, double *b, double *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API multiply_broadcast_tensor_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API multiply_broadcast_tensor_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API multiply_broadcast_tensor_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m,
                                                        int64_t n);
SDN_DLL_EXPORT_API multiply_broadcast_tensor_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API multiply_broadcast_tensor_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m,
                                                        int64_t n);
SDN_DLL_EXPORT_API multiply_broadcast_tensor_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API multiply_broadcast_tensor_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m,
                                                        int64_t n);
SDN_DLL_EXPORT_API multiply_broadcast_tensor_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n);

SDN_DLL_EXPORT_API divide_tensor_tensor_fp32(float *a, float *b, float *result, int64_t n);
SDN_DLL_EXPORT_API divide_tensor_tensor_fp64(double *a, double *b, double *result, int64_t n);
SDN_DLL_EXPORT_API divide_tensor_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t n);
SDN_DLL_EXPORT_API divide_tensor_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t n);
SDN_DLL_EXPORT_API divide_tensor_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t n);
SDN_DLL_EXPORT_API divide_tensor_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t n);
SDN_DLL_EXPORT_API divide_tensor_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t n);
SDN_DLL_EXPORT_API divide_tensor_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t n);
SDN_DLL_EXPORT_API divide_tensor_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t n);
SDN_DLL_EXPORT_API divide_tensor_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t n);
SDN_DLL_EXPORT_API divide_tensor_broadcast_tensor_fp32(float *a, float *b, float *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API divide_tensor_broadcast_tensor_fp64(double *a, double *b, double *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API divide_tensor_broadcast_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API divide_tensor_broadcast_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API divide_tensor_broadcast_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m,
                                                      int64_t n);
SDN_DLL_EXPORT_API divide_tensor_broadcast_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API divide_tensor_broadcast_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m,
                                                      int64_t n);
SDN_DLL_EXPORT_API divide_tensor_broadcast_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API divide_tensor_broadcast_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m,
                                                      int64_t n);
SDN_DLL_EXPORT_API divide_tensor_broadcast_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API divide_broadcast_tensor_tensor_fp32(float *a, float *b, float *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API divide_broadcast_tensor_tensor_fp64(double *a, double *b, double *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API divide_broadcast_tensor_tensor_u8(uint8_t *a, uint8_t *b, uint8_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API divide_broadcast_tensor_tensor_i8(int8_t *a, int8_t *b, int8_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API divide_broadcast_tensor_tensor_u16(uint16_t *a, uint16_t *b, uint16_t *result, int64_t m,
                                                      int64_t n);
SDN_DLL_EXPORT_API divide_broadcast_tensor_tensor_i16(int16_t *a, int16_t *b, int16_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API divide_broadcast_tensor_tensor_u32(uint32_t *a, uint32_t *b, uint32_t *result, int64_t m,
                                                      int64_t n);
SDN_DLL_EXPORT_API divide_broadcast_tensor_tensor_i32(int32_t *a, int32_t *b, int32_t *result, int64_t m, int64_t n);
SDN_DLL_EXPORT_API divide_broadcast_tensor_tensor_u64(uint64_t *a, uint64_t *b, uint64_t *result, int64_t m,
                                                      int64_t n);
SDN_DLL_EXPORT_API divide_broadcast_tensor_tensor_i64(int64_t *a, int64_t *b, int64_t *result, int64_t m, int64_t n);

SDN_DLL_EXPORT_API negate_tensor_fp32(float *a, float *result, int64_t n);
SDN_DLL_EXPORT_API negate_tensor_fp64(double *a, double *result, int64_t n);
SDN_DLL_EXPORT_API negate_tensor_i8(int8_t *a, int8_t *result, int64_t n);
SDN_DLL_EXPORT_API negate_tensor_i16(int16_t *a, int16_t *result, int64_t n);
SDN_DLL_EXPORT_API negate_tensor_i32(int32_t *a, int32_t *result, int64_t n);
SDN_DLL_EXPORT_API negate_tensor_i64(int64_t *a, int64_t *result, int64_t n);

SDN_DLL_EXPORT_API abs_tensor_fp32(float *a, float *result, int64_t n);
SDN_DLL_EXPORT_API abs_tensor_fp64(double *a, double *result, int64_t n);
SDN_DLL_EXPORT_API abs_tensor_i8(int8_t *a, int8_t *result, int64_t n);
SDN_DLL_EXPORT_API abs_tensor_i16(int16_t *a, int16_t *result, int64_t n);
SDN_DLL_EXPORT_API abs_tensor_i32(int32_t *a, int32_t *result, int64_t n);
SDN_DLL_EXPORT_API abs_tensor_i64(int64_t *a, int64_t *result, int64_t n);

SDN_DLL_EXPORT_API sqrt_tensor_fp32(float *a, float *result, int64_t n);
SDN_DLL_EXPORT_API sqrt_tensor_fp64(double *a, double *result, int64_t n);
SDN_DLL_EXPORT_API sqrt_tensor_u8(uint8_t *a, uint8_t *result, int64_t n);
SDN_DLL_EXPORT_API sqrt_tensor_i8(int8_t *a, int8_t *result, int64_t n);
SDN_DLL_EXPORT_API sqrt_tensor_u16(uint16_t *a, uint16_t *result, int64_t n);
SDN_DLL_EXPORT_API sqrt_tensor_i16(int16_t *a, int16_t *result, int64_t n);
SDN_DLL_EXPORT_API sqrt_tensor_u32(uint32_t *a, uint32_t *result, int64_t n);
SDN_DLL_EXPORT_API sqrt_tensor_i32(int32_t *a, int32_t *result, int64_t n);
SDN_DLL_EXPORT_API sqrt_tensor_u64(uint64_t *a, uint64_t *result, int64_t n);
SDN_DLL_EXPORT_API sqrt_tensor_i64(int64_t *a, int64_t *result, int64_t n);


#endif //SCI_NET_NATIVE_ARITHMETIC_H
