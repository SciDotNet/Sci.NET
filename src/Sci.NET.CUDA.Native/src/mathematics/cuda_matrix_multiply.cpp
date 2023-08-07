//
// Created by reece on 01/08/2023.
//

#include "cuda_matrix_multiply.h"
#include "kernels/matrix_multiply_kernels.cuh"


static cublasHandle_t handle;

inline cublasHandle_t get_handle() {
    if (handle == nullptr) {
        cublasCreate(&handle);
    }

    return handle;
}

sdnApiStatusCode matrix_multiply_fp32(float *left,
                                      float *right,
                                      float *result,
                                      int32_t left_rows,
                                      int32_t left_cols,
                                      int32_t right_cols) {
    float alpha = 1.0f;
    float beta = 0.0f;

    auto status = cublasSgemm_v2(get_handle(),
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 left_rows,
                                 right_cols,
                                 left_cols,
                                 &alpha,
                                 left,
                                 left_rows,
                                 right,
                                 right_cols,
                                 &beta,
                                 result,
                                 left_rows);

    return guard_cublas_status(status);
}

sdnApiStatusCode matrix_multiply_fp64(double *left,
                                      double *right,
                                      double *result,
                                      int32_t left_rows,
                                      int32_t left_cols,
                                      int32_t right_cols) {
    double alpha = 1.0f;
    double beta = 0.0f;

    auto status = cublasDgemm_v2(get_handle(),
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 left_rows,
                                 right_cols,
                                 left_cols,
                                 &alpha,
                                 left,
                                 left_rows,
                                 right,
                                 right_cols,
                                 &beta,
                                 result,
                                 left_rows);

    return guard_cublas_status(status);
}

sdnApiStatusCode matrix_multiply_u8(uint8_t *left,
                                    uint8_t *right,
                                    uint8_t *result,
                                    int32_t left_rows,
                                    int32_t left_cols,
                                    int32_t right_cols) {
    auto success = matrix_multiply_u8_invoke(left,
                                             right,
                                             result,
                                             left_rows,
                                             left_cols,
                                             right_cols);

    return success ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode matrix_multiply_i8(int8_t *left,
                                    int8_t *right,
                                    int8_t *result,
                                    int32_t left_rows,
                                    int32_t left_cols,
                                    int32_t right_cols) {

    auto success = matrix_multiply_i8_invoke(left,
                                             right,
                                             result,
                                             left_rows,
                                             left_cols,
                                             right_cols);

    return success ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode matrix_multiply_u16(uint16_t *left,
                                     uint16_t *right,
                                     uint16_t *result,
                                     int32_t left_rows,
                                     int32_t left_cols,
                                     int32_t right_cols) {

    auto success = matrix_multiply_u16_invoke(left,
                                              right,
                                              result,
                                              left_rows,
                                              left_cols,
                                              right_cols);

    return success ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode matrix_multiply_i16(int16_t *left,
                                     int16_t *right,
                                     int16_t *result,
                                     int32_t left_rows,
                                     int32_t left_cols,
                                     int32_t right_cols) {
    auto success = matrix_multiply_i16_invoke(left,
                                              right,
                                              result,
                                              left_rows,
                                              left_cols,
                                              right_cols);

    return success ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode matrix_multiply_u32(uint32_t *left,
                                     uint32_t *right,
                                     uint32_t *result,
                                     int32_t left_rows,
                                     int32_t left_cols,
                                     int32_t right_cols) {
    auto success = matrix_multiply_u32_invoke(left,
                                              right,
                                              result,
                                              left_rows,
                                              left_cols,
                                              right_cols);

    return success ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode matrix_multiply_i32(int32_t *left,
                                     int32_t *right,
                                     int32_t *result,
                                     int32_t left_rows,
                                     int32_t left_cols,
                                     int32_t right_cols) {
    auto success = matrix_multiply_i32_invoke(left,
                                              right,
                                              result,
                                              left_rows,
                                              left_cols,
                                              right_cols);

    return success ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode matrix_multiply_u64(uint64_t *left,
                                     uint64_t *right,
                                     uint64_t *result,
                                     int32_t left_rows,
                                     int32_t left_cols,
                                     int32_t right_cols) {
    auto success = matrix_multiply_u64_invoke(left,
                                              right,
                                              result,
                                              left_rows,
                                              left_cols,
                                              right_cols);

    return success ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

sdnApiStatusCode matrix_multiply_i64(int64_t *left,
                                     int64_t *right,
                                     int64_t *result,
                                     int32_t left_rows,
                                     int32_t left_cols,
                                     int32_t right_cols) {
    auto success = matrix_multiply_i64_invoke(left,
                                              right,
                                              result,
                                              left_rows,
                                              left_cols,
                                              right_cols);

    return success ? sdnApiStatusCode::sdnSuccess : sdnApiStatusCode::sdnInternalError;
}

