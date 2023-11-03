//
// Created by reece on 01/08/2023.
//

#ifndef SCI_NET_NATIVE_INNER_PRODUCT_KERNELS_CUH
#define SCI_NET_NATIVE_INNER_PRODUCT_KERNELS_CUH

bool inner_product_fp32_invoke(float *a, float *b, float *c, int32_t n);

bool inner_product_fp64_invoke(double *a, double *b, double *c, int32_t n);

bool inner_product_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *c, int32_t n);

bool inner_product_i8_invoke(int8_t *a, int8_t *b, int8_t *c, int32_t n);

bool inner_product_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *c, int32_t n);

bool inner_product_i16_invoke(int16_t *a, int16_t *b, int16_t *c, int32_t n);

bool inner_product_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *c, int32_t n);

bool inner_product_i32_invoke(int32_t *a, int32_t *b, int32_t *c, int32_t n);

bool inner_product_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *c, int32_t n);

bool inner_product_i64_invoke(int64_t *a, int64_t *b, int64_t *c, int32_t n);

#endif //SCI_NET_NATIVE_INNER_PRODUCT_KERNELS_CUH
