//
// Created by reece on 01/08/2023.
//

#ifndef SCI_NET_NATIVE_INNER_PRODUCT_H
#define SCI_NET_NATIVE_INNER_PRODUCT_H

#include "api.h"

SDN_DLL_EXPORT_API inner_product_fp32(float *a, float *b, float *c, int32_t n);

SDN_DLL_EXPORT_API inner_product_fp64(double *a, double *b, double *c, int32_t n);

SDN_DLL_EXPORT_API inner_product_u8(uint8_t *a, uint8_t *b, uint8_t *c, int32_t n);

SDN_DLL_EXPORT_API inner_product_i8(int8_t *a, int8_t *b, int8_t *c, int32_t n);

SDN_DLL_EXPORT_API inner_product_u16(uint16_t *a, uint16_t *b, uint16_t *c, int32_t n);

SDN_DLL_EXPORT_API inner_product_i16(int16_t *a, int16_t *b, int16_t *c, int32_t n);

SDN_DLL_EXPORT_API inner_product_u32(uint32_t *a, uint32_t *b, uint32_t *c, int32_t n);

SDN_DLL_EXPORT_API inner_product_i32(int32_t *a, int32_t *b, int32_t *c, int32_t n);

SDN_DLL_EXPORT_API inner_product_u64(uint64_t *a, uint64_t *b, uint64_t *c, int32_t n);

SDN_DLL_EXPORT_API inner_product_i64(int64_t *a, int64_t *b, int64_t *c, int32_t n);

#endif //SCI_NET_NATIVE_INNER_PRODUCT_H
