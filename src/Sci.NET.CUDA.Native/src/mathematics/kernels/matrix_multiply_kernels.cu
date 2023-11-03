//
// Created by reece on 01/08/2023.
//

#include "matrix_multiply_kernels.cuh"
#include "mma.h"

__global__ void matrix_multiply_bf16_kernel(nv_bfloat16 *left,
                                            nv_bfloat16 *right,
                                            nv_bfloat16 *result,
                                            int32_t left_rows,
                                            int32_t left_cols,
                                            int32_t right_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < left_rows && col < right_cols) {
        nv_bfloat16 sum = (nv_bfloat16)0.0f;

        for (int i = 0; i < left_cols; i++) {
            sum = sum + (nv_bfloat16)(left[row * left_cols + i] * right[i * right_cols + col]);
        }

        result[row * right_cols + col] = sum;
    }
}

__global__ void matrix_multiply_u8_kernel(uint8_t *left,
                                          uint8_t *right,
                                          uint8_t *result,
                                          int32_t left_rows,
                                          int32_t left_cols,
                                          int32_t right_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < left_rows && col < right_cols) {
        uint8_t sum = 0;

        for (int i = 0; i < left_cols; i++) {
            sum += left[row * left_cols + i] * right[i * right_cols + col];
        }

        result[row * right_cols + col] = sum;
    }
}

__global__ void matrix_multiply_i8_kernel(int8_t *left,
                                          int8_t *right,
                                          int8_t *result,
                                          int32_t left_rows,
                                          int32_t left_cols,
                                          int32_t right_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < left_rows && col < right_cols) {
        int8_t sum = 0;

        for (int i = 0; i < left_cols; i++) {
            sum += left[row * left_cols + i] * right[i * right_cols + col];
        }

        result[row * right_cols + col] = sum;
    }
}

__global__ void matrix_multiply_u16_kernel(uint16_t *left,
                                           uint16_t *right,
                                           uint16_t *result,
                                           int32_t left_rows,
                                           int32_t left_cols,
                                           int32_t right_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < left_rows && col < right_cols) {
        uint16_t sum = 0;

        for (int i = 0; i < left_cols; i++) {
            sum += left[row * left_cols + i] * right[i * right_cols + col];
        }

        result[row * right_cols + col] = sum;
    }
}

__global__ void matrix_multiply_i16_kernel(int16_t *left,
                                           int16_t *right,
                                           int16_t *result,
                                           int32_t left_rows,
                                           int32_t left_cols,
                                           int32_t right_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < left_rows && col < right_cols) {
        int16_t sum = 0;

        for (int i = 0; i < left_cols; i++) {
            sum += left[row * left_cols + i] * right[i * right_cols + col];
        }

        result[row * right_cols + col] = sum;
    }
}

__global__ void matrix_multiply_u32_kernel(uint32_t *left,
                                           uint32_t *right,
                                           uint32_t *result,
                                           int32_t left_rows,
                                           int32_t left_cols,
                                           int32_t right_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < left_rows && col < right_cols) {
        uint32_t sum = 0;

        for (int i = 0; i < left_cols; i++) {
            sum += left[row * left_cols + i] * right[i * right_cols + col];
        }

        result[row * right_cols + col] = sum;
    }
}

__global__ void matrix_multiply_i32_kernel(int32_t *left,
                                           int32_t *right,
                                           int32_t *result,
                                           int32_t left_rows,
                                           int32_t left_cols,
                                           int32_t right_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < left_rows && col < right_cols) {
        int32_t sum = 0;

        for (int i = 0; i < left_cols; i++) {
            sum += left[row * left_cols + i] * right[i * right_cols + col];
        }

        result[row * right_cols + col] = sum;
    }
}

__global__ void matrix_multiply_u64_kernel(uint64_t *left,
                                           uint64_t *right,
                                           uint64_t *result,
                                           int32_t left_rows,
                                           int32_t left_cols,
                                           int32_t right_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < left_rows && col < right_cols) {
        uint64_t sum = 0;

        for (int i = 0; i < left_cols; i++) {
            sum += left[row * left_cols + i] * right[i * right_cols + col];
        }

        result[row * right_cols + col] = sum;
    }
}

__global__ void matrix_multiply_i64_kernel(int64_t *left,
                                           int64_t *right,
                                           int64_t *result,
                                           int32_t left_rows,
                                           int32_t left_cols,
                                           int32_t right_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < left_rows && col < right_cols) {
        int64_t sum = 0;

        for (int i = 0; i < left_cols; i++) {
            sum += left[row * left_cols + i] * right[i * right_cols + col];
        }

        result[row * right_cols + col] = sum;
    }
}

bool matrix_multiply_bf16_invoke(nv_bfloat16 *left,
                                 nv_bfloat16 *right,
                                 nv_bfloat16 *result,
                                 int32_t left_rows,
                                 int32_t left_cols,
                                 int32_t right_cols) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((right_cols + dimBlock.x - 1) / dimBlock.x,
                 (left_rows + dimBlock.y - 1) / dimBlock.y);

    matrix_multiply_bf16_kernel<<<dimGrid, dimBlock>>>(left, right, result, left_rows, left_cols, right_cols);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}

bool matrix_multiply_u8_invoke(uint8_t *left,
                               uint8_t *right,
                               uint8_t *result,
                               int32_t left_rows,
                               int32_t left_cols,
                               int32_t right_cols) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((right_cols + dimBlock.x - 1) / dimBlock.x,
                 (left_rows + dimBlock.y - 1) / dimBlock.y);

    matrix_multiply_u8_kernel<<<dimGrid, dimBlock>>>(left, right, result, left_rows, left_cols, right_cols);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}

bool matrix_multiply_i8_invoke(int8_t *left,
                               int8_t *right,
                               int8_t *result,
                               int32_t left_rows,
                               int32_t left_cols,
                               int32_t right_cols) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((right_cols + dimBlock.x - 1) / dimBlock.x,
                 (left_rows + dimBlock.y - 1) / dimBlock.y);

    matrix_multiply_i8_kernel<<<dimGrid, dimBlock>>>(left, right, result, left_rows, left_cols, right_cols);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}

bool matrix_multiply_u16_invoke(uint16_t *left,
                                uint16_t *right,
                                uint16_t *result,
                                int32_t left_rows,
                                int32_t left_cols,
                                int32_t right_cols) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((right_cols + dimBlock.x - 1) / dimBlock.x,
                 (left_rows + dimBlock.y - 1) / dimBlock.y);

    matrix_multiply_u16_kernel<<<dimGrid, dimBlock>>>(left, right, result, left_rows, left_cols, right_cols);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}

bool matrix_multiply_i16_invoke(int16_t *left,
                                int16_t *right,
                                int16_t *result,
                                int32_t left_rows,
                                int32_t left_cols,
                                int32_t right_cols) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((right_cols + dimBlock.x - 1) / dimBlock.x,
                 (left_rows + dimBlock.y - 1) / dimBlock.y);

    matrix_multiply_i16_kernel<<<dimGrid, dimBlock>>>(left, right, result, left_rows, left_cols, right_cols);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}

bool matrix_multiply_u32_invoke(uint32_t *left,
                                uint32_t *right,
                                uint32_t *result,
                                int32_t left_rows,
                                int32_t left_cols,
                                int32_t right_cols) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((right_cols + dimBlock.x - 1) / dimBlock.x,
                 (left_rows + dimBlock.y - 1) / dimBlock.y);

    matrix_multiply_u32_kernel<<<dimGrid, dimBlock>>>(left, right, result, left_rows, left_cols, right_cols);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}

bool matrix_multiply_i32_invoke(int32_t *left,
                                int32_t *right,
                                int32_t *result,
                                int32_t left_rows,
                                int32_t left_cols,
                                int32_t right_cols) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((right_cols + dimBlock.x - 1) / dimBlock.x,
                 (left_rows + dimBlock.y - 1) / dimBlock.y);

    matrix_multiply_i32_kernel<<<dimGrid, dimBlock>>>(left, right, result, left_rows, left_cols, right_cols);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}

bool matrix_multiply_u64_invoke(uint64_t *left,
                                uint64_t *right,
                                uint64_t *result,
                                int32_t left_rows,
                                int32_t left_cols,
                                int32_t right_cols) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((right_cols + dimBlock.x - 1) / dimBlock.x,
                 (left_rows + dimBlock.y - 1) / dimBlock.y);

    matrix_multiply_u64_kernel<<<dimGrid, dimBlock>>>(left, right, result, left_rows, left_cols, right_cols);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}

bool matrix_multiply_i64_invoke(int64_t *left,
                                int64_t *right,
                                int64_t *result,
                                int32_t left_rows,
                                int32_t left_cols,
                                int32_t right_cols) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((right_cols + dimBlock.x - 1) / dimBlock.x,
                 (left_rows + dimBlock.y - 1) / dimBlock.y);

    matrix_multiply_i64_kernel<<<dimGrid, dimBlock>>>(left, right, result, left_rows, left_cols, right_cols);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}
