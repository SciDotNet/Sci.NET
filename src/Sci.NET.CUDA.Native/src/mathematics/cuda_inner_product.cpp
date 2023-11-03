//
// Created by reece on 01/08/2023.
//


#include "cuda_inner_product.h"
#include "kernels/inner_product_kernels.cuh"


sdnApiStatusCode inner_product_fp32(float *a, float *b, float *c, int32_t n) {

    auto result = inner_product_fp32_invoke(a, b, c, n);

    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode inner_product_fp64(double *a, double *b, double *c, int32_t n) {

    auto result = inner_product_fp64_invoke(a, b, c, n);

    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode inner_product_u8(uint8_t *a, uint8_t *b, uint8_t *c, int32_t n) {

    auto result = inner_product_u8_invoke(a, b, c, n);

    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode inner_product_i8(int8_t *a, int8_t *b, int8_t *c, int32_t n) {

    auto result = inner_product_i8_invoke(a, b, c, n);

    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode inner_product_u16(uint16_t *a, uint16_t *b, uint16_t *c, int32_t n) {

    auto result = inner_product_u16_invoke(a, b, c, n);

    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode inner_product_i16(int16_t *a, int16_t *b, int16_t *c, int32_t n) {

    auto result = inner_product_i16_invoke(a, b, c, n);

    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode inner_product_u32(uint32_t *a, uint32_t *b, uint32_t *c, int32_t n) {

    auto result = inner_product_u32_invoke(a, b, c, n);

    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode inner_product_i32(int32_t *a, int32_t *b, int32_t *c, int32_t n) {

    auto result = inner_product_i32_invoke(a, b, c, n);

    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode inner_product_u64(uint64_t *a, uint64_t *b, uint64_t *c, int32_t n) {

    auto result = inner_product_u64_invoke(a, b, c, n);

    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode inner_product_i64(int64_t *a, int64_t *b, int64_t *c, int32_t n) {

    auto result = inner_product_i64_invoke(a, b, c, n);

    return result ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}
