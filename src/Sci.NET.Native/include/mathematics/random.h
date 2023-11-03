//
// Created by reece on 02/08/2023.
//

#ifndef SCI_NET_NATIVE_RANDOM_H
#define SCI_NET_NATIVE_RANDOM_H

#include "api.h"

SDN_DLL_EXPORT_API random_uniform_fp32(float *dst, float min, float max, size_t size, long seed);
SDN_DLL_EXPORT_API random_uniform_fp64(double *dst, double min, double max, size_t size, long seed);
SDN_DLL_EXPORT_API random_uniform_u8(uint8_t *dst, uint8_t min, uint8_t max, size_t size, long seed);
SDN_DLL_EXPORT_API random_uniform_i8(int8_t *dst, int8_t min, int8_t max, size_t size, long seed);
SDN_DLL_EXPORT_API random_uniform_u16(uint16_t *dst, uint16_t min, uint16_t max, size_t size, long seed);
SDN_DLL_EXPORT_API random_uniform_i16(int16_t *dst, int16_t min, int16_t max, size_t size, long seed);
SDN_DLL_EXPORT_API random_uniform_u32(uint32_t *dst, uint32_t min, uint32_t max, size_t size, long seed);
SDN_DLL_EXPORT_API random_uniform_i32(int32_t *dst, int32_t min, int32_t max, size_t size, long seed);
SDN_DLL_EXPORT_API random_uniform_u64(uint64_t *dst, uint64_t min, uint64_t max, size_t size, long seed);
SDN_DLL_EXPORT_API random_uniform_i64(int64_t *dst, int64_t min, int64_t max, size_t size, long seed);

#endif //SCI_NET_NATIVE_RANDOM_H
