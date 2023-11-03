//
// Created by reece on 01/08/2023.
//

#include "inner_product_kernels.cuh"

__global__ void inner_product_kernel_fp32(float *a, float *b, float *c, int32_t n) {

    __shared__ float cache[256]; // Shared memory for each block, adjust the size as needed

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + tid;

    cache[tid] = (i < n) ? a[i] * b[i] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    // Store the final result in global memory
    if (tid == 0) {
        c[blockIdx.x] = cache[0];
    }
}

__global__ void inner_product_kernel_fp64(double *a, double *b, double *c, int32_t n) {
    __shared__ double cache[256]; // Shared memory for each block, adjust the size as needed

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + tid;

    cache[tid] = (i < n) ? a[i] * b[i] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    // Store the final result in global memory
    if (tid == 0) {
        c[blockIdx.x] = cache[0];
    }
}

__global__ void inner_product_kernel_u8(uint8_t *a, uint8_t *b, uint8_t *c, int32_t n) {
    __shared__ uint8_t cache[256]; // Shared memory for each block, adjust the size as needed

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + tid;

    cache[tid] = (i < n) ? a[i] * b[i] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    // Store the final result in global memory
    if (tid == 0) {
        c[blockIdx.x] = cache[0];
    }
}

__global__ void inner_product_kernel_i8(int8_t *a, int8_t *b, int8_t *c, int32_t n) {
    __shared__ int8_t cache[256]; // Shared memory for each block, adjust the size as needed

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + tid;

    cache[tid] = (i < n) ? a[i] * b[i] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    // Store the final result in global memory
    if (tid == 0) {
        c[blockIdx.x] = cache[0];
    }
}

__global__ void inner_product_kernel_u16(uint16_t *a, uint16_t *b, uint16_t *c, int32_t n) {
    __shared__ uint16_t cache[256]; // Shared memory for each block, adjust the size as needed

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + tid;

    cache[tid] = (i < n) ? a[i] * b[i] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    // Store the final result in global memory
    if (tid == 0) {
        c[blockIdx.x] = cache[0];
    }
}

__global__ void inner_product_kernel_i16(int16_t *a, int16_t *b, int16_t *c, int32_t n) {
    __shared__ int16_t cache[256]; // Shared memory for each block, adjust the size as needed

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + tid;

    cache[tid] = (i < n) ? a[i] * b[i] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    // Store the final result in global memory
    if (tid == 0) {
        c[blockIdx.x] = cache[0];
    }
}

__global__ void inner_product_kernel_u32(uint32_t *a, uint32_t *b, uint32_t *c, int32_t n) {
    __shared__ uint32_t cache[256]; // Shared memory for each block, adjust the size as needed

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + tid;

    cache[tid] = (i < n) ? a[i] * b[i] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    // Store the final result in global memory
    if (tid == 0) {
        c[blockIdx.x] = cache[0];
    }
}

__global__ void inner_product_kernel_i32(int32_t *a, int32_t *b, int32_t *c, int32_t n) {
    __shared__ int32_t cache[256]; // Shared memory for each block, adjust the size as needed

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + tid;

    cache[tid] = (i < n) ? a[i] * b[i] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    // Store the final result in global memory
    if (tid == 0) {
        c[blockIdx.x] = cache[0];
    }
}

__global__ void inner_product_kernel_u64(uint64_t *a, uint64_t *b, uint64_t *c, int32_t n) {
    __shared__ uint64_t cache[256]; // Shared memory for each block, adjust the size as needed

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + tid;

    cache[tid] = (i < n) ? a[i] * b[i] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    // Store the final result in global memory
    if (tid == 0) {
        c[blockIdx.x] = cache[0];
    }
}

__global__ void inner_product_kernel_i64(int64_t *a, int64_t *b, int64_t *c, int32_t n) {
    __shared__ int64_t cache[256]; // Shared memory for each block, adjust the size as needed

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + tid;

    cache[tid] = (i < n) ? a[i] * b[i] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    // Store the final result in global memory
    if (tid == 0) {
        c[blockIdx.x] = cache[0];
    }
}

bool inner_product_fp32_invoke(float *a, float *b, float *c, int32_t n) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    inner_product_kernel_fp32<<<num_blocks, block_size>>>(a, b, c, n);
    cudaDeviceSynchronize();

    return cudaGetLastError() == cudaSuccess;
}

bool inner_product_fp64_invoke(double *a, double *b, double *c, int32_t n) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    inner_product_kernel_fp64<<<num_blocks, block_size>>>(a, b, c, n);
    cudaDeviceSynchronize();

    return cudaGetLastError() == cudaSuccess;
}

bool inner_product_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *c, int32_t n) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    inner_product_kernel_u8<<<num_blocks, block_size>>>(a, b, c, n);
    cudaDeviceSynchronize();

    return cudaGetLastError() == cudaSuccess;
}

bool inner_product_i8_invoke(int8_t *a, int8_t *b, int8_t *c, int32_t n) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    inner_product_kernel_i8<<<num_blocks, block_size>>>(a, b, c, n);
    cudaDeviceSynchronize();

    return cudaGetLastError() == cudaSuccess;
}

bool inner_product_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *c, int32_t n) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    inner_product_kernel_u16<<<num_blocks, block_size>>>(a, b, c, n);
    cudaDeviceSynchronize();

    return cudaGetLastError() == cudaSuccess;
}

bool inner_product_i16_invoke(int16_t *a, int16_t *b, int16_t *c, int32_t n) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    inner_product_kernel_i16<<<num_blocks, block_size>>>(a, b, c, n);
    cudaDeviceSynchronize();

    return cudaGetLastError() == cudaSuccess;
}

bool inner_product_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *c, int32_t n) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    inner_product_kernel_u32<<<num_blocks, block_size>>>(a, b, c, n);
    cudaDeviceSynchronize();

    return cudaGetLastError() == cudaSuccess;
}

bool inner_product_i32_invoke(int32_t *a, int32_t *b, int32_t *c, int32_t n) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    inner_product_kernel_i32<<<num_blocks, block_size>>>(a, b, c, n);
    cudaDeviceSynchronize();

    return cudaGetLastError() == cudaSuccess;
}

bool inner_product_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *c, int32_t n) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    inner_product_kernel_u64<<<num_blocks, block_size>>>(a, b, c, n);
    cudaDeviceSynchronize();

    return cudaGetLastError() == cudaSuccess;
}

bool inner_product_i64_invoke(int64_t *a, int64_t *b, int64_t *c, int32_t n) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    inner_product_kernel_i64<<<num_blocks, block_size>>>(a, b, c, n);
    cudaDeviceSynchronize();

    return cudaGetLastError() == cudaSuccess;
}
