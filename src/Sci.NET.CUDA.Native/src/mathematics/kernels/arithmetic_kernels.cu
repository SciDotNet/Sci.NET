
#include "arithmetic_kernels.cuh"

#define BLOCK_SIZE 256
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void add_tensor_tensor_bf16_kernel(nv_bfloat16 *a, nv_bfloat16 *b, nv_bfloat16 *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_tensor_tensor_fp32_kernel(float *a, float *b, float *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_tensor_tensor_fp64_kernel(double *a, double *b, double *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_tensor_tensor_u8_kernel(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_tensor_tensor_i8_kernel(int8_t *a, int8_t *b, int8_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_tensor_tensor_u16_kernel(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_tensor_tensor_i16_kernel(int16_t *a, int16_t *b, int16_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_tensor_tensor_u32_kernel(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_tensor_tensor_i32_kernel(int32_t *a, int32_t *b, int32_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_tensor_tensor_u64_kernel(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_tensor_tensor_i64_kernel(int64_t *a, int64_t *b, int64_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_tensor_broadcast_tensor_fp32_kernel(float *a, float *b, float *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] + b[j];
    }
}

__global__ void add_tensor_broadcast_tensor_fp64_kernel(double *a, double *b, double *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] + b[j];
    }
}

__global__ void add_tensor_broadcast_tensor_u8_kernel(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] + b[j];
    }
}

__global__ void add_tensor_broadcast_tensor_i8_kernel(int8_t *a, int8_t *b, int8_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] + b[j];
    }
}

__global__ void
add_tensor_broadcast_tensor_u16_kernel(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] + b[j];
    }
}

__global__ void add_tensor_broadcast_tensor_i16_kernel(int16_t *a, int16_t *b, int16_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] + b[j];
    }
}

__global__ void
add_tensor_broadcast_tensor_u32_kernel(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] + b[j];
    }
}

__global__ void add_tensor_broadcast_tensor_i32_kernel(int32_t *a, int32_t *b, int32_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] + b[j];
    }
}

__global__ void
add_tensor_broadcast_tensor_u64_kernel(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] + b[j];
    }
}

__global__ void add_tensor_broadcast_tensor_i64_kernel(int64_t *a, int64_t *b, int64_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] + b[j];
    }
}

__global__ void add_broadcast_tensor_tensor_fp32_kernel(float *a, float *b, float *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < n && i < m) {
        c[(i * n) + j] = a[i] + b[(i * n) + j];
    }
}

__global__ void add_broadcast_tensor_tensor_fp64_kernel(double *a, double *b, double *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < n && i < m) {
        c[(i * n) + j] = a[i] + b[(i * n) + j];
    }
}

__global__ void add_broadcast_tensor_tensor_u8_kernel(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < n && i < m) {
        c[(i * n) + j] = a[i] + b[(i * n) + j];
    }
}

__global__ void add_broadcast_tensor_tensor_i8_kernel(int8_t *a, int8_t *b, int8_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < n && i < m) {
        c[(i * n) + j] = a[i] + b[(i * n) + j];
    }
}

__global__ void
add_broadcast_tensor_tensor_u16_kernel(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < n && i < m) {
        c[(i * n) + j] = a[i] + b[(i * n) + j];
    }
}

__global__ void add_broadcast_tensor_tensor_i16_kernel(int16_t *a, int16_t *b, int16_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < n && i < m) {
        c[(i * n) + j] = a[i] + b[(i * n) + j];
    }
}

__global__ void
add_broadcast_tensor_tensor_u32_kernel(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < n && i < m) {
        c[(i * n) + j] = a[i] + b[(i * n) + j];
    }
}

__global__ void add_broadcast_tensor_tensor_i32_kernel(int32_t *a, int32_t *b, int32_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < n && i < m) {
        c[(i * n) + j] = a[i] + b[(i * n) + j];
    }
}

__global__ void
add_broadcast_tensor_tensor_u64_kernel(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < n && i < m) {
        c[(i * n) + j] = a[i] + b[(i * n) + j];
    }
}

__global__ void add_broadcast_tensor_tensor_i64_kernel(int64_t *a, int64_t *b, int64_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < n && i < m) {
        c[(i * n) + j] = a[i] + b[(i * n) + j];
    }
}

__global__ void subtract_tensor_tensor_fp32_kernel(float *a, float *b, float *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

__global__ void subtract_tensor_tensor_fp64_kernel(double *a, double *b, double *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

__global__ void subtract_tensor_tensor_u8_kernel(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

__global__ void subtract_tensor_tensor_i8_kernel(int8_t *a, int8_t *b, int8_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

__global__ void subtract_tensor_tensor_u16_kernel(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

__global__ void subtract_tensor_tensor_i16_kernel(int16_t *a, int16_t *b, int16_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

__global__ void subtract_tensor_tensor_u32_kernel(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

__global__ void subtract_tensor_tensor_i32_kernel(int32_t *a, int32_t *b, int32_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

__global__ void subtract_tensor_tensor_u64_kernel(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

__global__ void subtract_tensor_tensor_i64_kernel(int64_t *a, int64_t *b, int64_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

__global__ void subtract_tensor_broadcast_tensor_fp32_kernel(float *a, float *b, float *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] - b[j];
    }
}

__global__ void subtract_tensor_broadcast_tensor_fp64_kernel(double *a, double *b, double *c, int64_t n,
                                                      int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] - b[j];
    }
}

__global__ void subtract_tensor_broadcast_tensor_u8_kernel(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n,
                                                    int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] - b[j];
    }
}

__global__ void subtract_tensor_broadcast_tensor_i8_kernel(int8_t *a, int8_t *b, int8_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] - b[j];
    }
}

__global__ void subtract_tensor_broadcast_tensor_u16_kernel(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] - b[j];
    }
}

__global__ void subtract_tensor_broadcast_tensor_i16_kernel(int16_t *a, int16_t *b, int16_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] - b[j];
    }
}

__global__ void subtract_tensor_broadcast_tensor_u32_kernel(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] - b[j];
    }
}

__global__ void subtract_tensor_broadcast_tensor_i32_kernel(int32_t *a, int32_t *b, int32_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] - b[j];
    }
}

__global__ void subtract_tensor_broadcast_tensor_u64_kernel(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] - b[j];
    }
}

__global__ void subtract_tensor_broadcast_tensor_i64_kernel(int64_t *a, int64_t *b, int64_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < m) {
        c[(i * n) + j] = a[(i * n) + j] - b[j];
    }
}

__global__ void subtract_broadcast_tensor_tensor_fp32_kernel(float *a, float *b, float *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < n && i < m) {
        c[(i * n) + j] = a[i] - b[(i * n) + j];
    }
}

__global__ void subtract_broadcast_tensor_tensor_fp64_kernel(double *a, double *b, double *c, int64_t n,
                                                      int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < n && i < m) {
        c[(i * n) + j] = a[i] - b[(i * n) + j];
    }
}

__global__ void subtract_broadcast_tensor_tensor_u8_kernel(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n,
                                                    int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < n && i < m) {
        c[(i * n) + j] = a[i] - b[(i * n) + j];
    }
}

__global__ void subtract_broadcast_tensor_tensor_i8_kernel(int8_t *a, int8_t *b, int8_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < n && i < m) {
        c[(i * n) + j] = a[i] - b[(i * n) + j];
    }
}

__global__ void subtract_broadcast_tensor_tensor_u16_kernel(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < n && i < m) {
        c[(i * n) + j] = a[i] - b[(i * n) + j];
    }
}

__global__ void subtract_broadcast_tensor_tensor_i16_kernel(int16_t *a, int16_t *b, int16_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < n && i < m) {
        c[(i * n) + j] = a[i] - b[(i * n) + j];
    }
}

__global__ void subtract_broadcast_tensor_tensor_u32_kernel(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < n && i < m) {
        c[(i * n) + j] = a[i] - b[(i * n) + j];
    }
}

__global__ void subtract_broadcast_tensor_tensor_i32_kernel(int32_t *a, int32_t *b, int32_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < n && i < m) {
        c[(i * n) + j] = a[i] - b[(i * n) + j];
    }
}

__global__ void subtract_broadcast_tensor_tensor_u64_kernel(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < n && i < m) {
        c[(i * n) + j] = a[i] - b[(i * n) + j];
    }
}

__global__ void subtract_broadcast_tensor_tensor_i64_kernel(int64_t *a, int64_t *b, int64_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < n && i < m) {
        c[(i * n) + j] = a[i] - b[(i * n) + j];
    }
}

__global__ void multiply_tensor_tensor_fp32_kernel(float *a, float *b, float *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

__global__ void multiply_tensor_tensor_fp64_kernel(double *a, double *b, double *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

__global__ void multiply_tensor_tensor_u8_kernel(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

__global__ void multiply_tensor_tensor_i8_kernel(int8_t *a, int8_t *b, int8_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

__global__ void multiply_tensor_tensor_u16_kernel(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

__global__ void multiply_tensor_tensor_i16_kernel(int16_t *a, int16_t *b, int16_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

__global__ void multiply_tensor_tensor_u32_kernel(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

__global__ void multiply_tensor_tensor_i32_kernel(int32_t *a, int32_t *b, int32_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

__global__ void multiply_tensor_tensor_u64_kernel(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

__global__ void multiply_tensor_tensor_i64_kernel(int64_t *a, int64_t *b, int64_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

__global__ void multiply_tensor_broadcast_tensor_fp32_kernel(float *a, float *b, float *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n) {
        c[(i * n) + j] = a[(i * n) + j] * b[j];
    }
}

__global__ void multiply_tensor_broadcast_tensor_fp64_kernel(double *a, double *b, double *c, int64_t n,
                                                      int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n) {
        c[(i * n) + j] = a[(i * n) + j] * b[j];
    }
}

__global__ void multiply_tensor_broadcast_tensor_u8_kernel(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n,
                                                    int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[(i * n) + j] * b[j];
    }
}

__global__ void multiply_tensor_broadcast_tensor_i8_kernel(int8_t *a, int8_t *b, int8_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[(i * n) + j] * b[j];
    }
}

__global__ void multiply_tensor_broadcast_tensor_u16_kernel(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[(i * n) + j] * b[j];
    }
}

__global__ void multiply_tensor_broadcast_tensor_i16_kernel(int16_t *a, int16_t *b, int16_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[(i * n) + j] * b[j];
    }
}

__global__ void multiply_tensor_broadcast_tensor_u32_kernel(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[(i * n) + j] * b[j];
    }
}

__global__ void multiply_tensor_broadcast_tensor_i32_kernel(int32_t *a, int32_t *b, int32_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[(i * n) + j] * b[j];
    }
}

__global__ void multiply_tensor_broadcast_tensor_u64_kernel(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[(i * n) + j] * b[j];
    }
}

__global__ void multiply_tensor_broadcast_tensor_i64_kernel(int64_t *a, int64_t *b, int64_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[(i * n) + j] * b[j];
    }
}

__global__ void multiply_broadcast_tensor_tensor_fp32_kernel(float *a, float *b, float *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n) {
        c[(i * n) + j] = a[i] * b[(i * n) + j];
    }
}

__global__ void multiply_broadcast_tensor_tensor_fp64_kernel(double *a, double *b, double *c, int64_t n,
                                                      int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n) {
        c[(i * n) + j] = a[i] * b[(i * n) + j];
    }
}

__global__ void multiply_broadcast_tensor_tensor_u8_kernel(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n,
                                                    int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] * b[(i * n) + j];
    }
}

__global__ void multiply_broadcast_tensor_tensor_i8_kernel(int8_t *a, int8_t *b, int8_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] * b[(i * n) + j];
    }
}

__global__ void multiply_broadcast_tensor_tensor_u16_kernel(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] * b[(i * n) + j];
    }
}

__global__ void multiply_broadcast_tensor_tensor_i16_kernel(int16_t *a, int16_t *b, int16_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] * b[(i * n) + j];
    }
}

__global__ void multiply_broadcast_tensor_tensor_u32_kernel(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] * b[(i * n) + j];
    }
}

__global__ void multiply_broadcast_tensor_tensor_i32_kernel(int32_t *a, int32_t *b, int32_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] * b[(i * n) + j];
    }
}

__global__ void multiply_broadcast_tensor_tensor_u64_kernel(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] * b[(i * n) + j];
    }
}

__global__ void multiply_broadcast_tensor_tensor_i64_kernel(int64_t *a, int64_t *b, int64_t *c, int64_t n,
                                                     int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] * b[(i * n) + j];
    }
}

__global__ void divide_tensor_tensor_fp32_kernel(float *a, float *b, float *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] / b[i];
    }
}

__global__ void divide_tensor_tensor_fp64_kernel(double *a, double *b, double *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] / b[i];
    }
}

__global__ void divide_tensor_tensor_u8_kernel(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] / b[i];
    }
}

__global__ void divide_tensor_tensor_i8_kernel(int8_t *a, int8_t *b, int8_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] / b[i];
    }
}

__global__ void divide_tensor_tensor_u16_kernel(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] / b[i];
    }
}

__global__ void divide_tensor_tensor_i16_kernel(int16_t *a, int16_t *b, int16_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] / b[i];
    }
}

__global__ void divide_tensor_tensor_u32_kernel(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] / b[i];
    }
}

__global__ void divide_tensor_tensor_i32_kernel(int32_t *a, int32_t *b, int32_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] / b[i];
    }
}

__global__ void divide_tensor_tensor_u64_kernel(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] / b[i];
    }
}

__global__ void divide_tensor_tensor_i64_kernel(int64_t *a, int64_t *b, int64_t *c, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] / b[i];
    }
}

__global__ void divide_tensor_broadcast_tensor_fp32_kernel(float *a, float *b, float *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n) {
        c[(i * n) + j] = a[(i * n) + j] / b[j];
    }
}

__global__ void divide_tensor_broadcast_tensor_fp64_kernel(double *a, double *b, double *c, int64_t n,
                                                    int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < n) {
        c[(i * n) + j] = a[(i * n) + j] / b[j];
    }
}

__global__ void divide_tensor_broadcast_tensor_u8_kernel(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n,
                                                  int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[(i * n) + j] / b[j];
    }
}

__global__ void divide_tensor_broadcast_tensor_i8_kernel(int8_t *a, int8_t *b, int8_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[(i * n) + j] / b[j];
    }
}

__global__ void divide_tensor_broadcast_tensor_u16_kernel(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n,
                                                   int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[(i * n) + j] / b[j];
    }
}

__global__ void divide_tensor_broadcast_tensor_i16_kernel(int16_t *a, int16_t *b, int16_t *c, int64_t n,
                                                   int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[(i * n) + j] / b[j];
    }
}

__global__ void divide_tensor_broadcast_tensor_u32_kernel(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n,
                                                   int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[(i * n) + j] / b[j];
    }
}

__global__ void divide_tensor_broadcast_tensor_i32_kernel(int32_t *a, int32_t *b, int32_t *c, int64_t n,
                                                   int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[(i * n) + j] / b[j];
    }
}

__global__ void divide_tensor_broadcast_tensor_u64_kernel(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n,
                                                   int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;


    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[(i * n) + j] / b[j];
    }
}

__global__ void divide_tensor_broadcast_tensor_i64_kernel(int64_t *a, int64_t *b, int64_t *c, int64_t n,
                                                   int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;


    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[(i * n) + j] / b[j];
    }
}

__global__ void divide_broadcast_tensor_tensor_fp32_kernel(float *a, float *b, float *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] / b[(i * n) + j];
    }
}

__global__ void divide_broadcast_tensor_tensor_fp64_kernel(double *a, double *b, double *c, int64_t n,
                                                    int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] / b[(i * n) + j];
    }
}

__global__ void divide_broadcast_tensor_tensor_u8_kernel(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n,
                                                  int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;


    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] / b[(i * n) + j];
    }
}

__global__ void divide_broadcast_tensor_tensor_i8_kernel(int8_t *a, int8_t *b, int8_t *c, int64_t n, int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;


    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] / b[(i * n) + j];
    }
}

__global__ void divide_broadcast_tensor_tensor_u16_kernel(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n,
                                                   int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;


    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] / b[(i * n) + j];
    }
}

__global__ void divide_broadcast_tensor_tensor_i16_kernel(int16_t *a, int16_t *b, int16_t *c, int64_t n,
                                                   int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;


    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] / b[(i * n) + j];
    }
}

__global__ void divide_broadcast_tensor_tensor_u32_kernel(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n,
                                                   int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;


    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] / b[(i * n) + j];
    }
}

__global__ void divide_broadcast_tensor_tensor_i32_kernel(int32_t *a, int32_t *b, int32_t *c, int64_t n,
                                                   int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] / b[(i * n) + j];
    }
}

__global__ void divide_broadcast_tensor_tensor_u64_kernel(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n,
                                                   int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] / b[(i * n) + j];
    }
}

__global__ void divide_broadcast_tensor_tensor_i64_kernel(int64_t *a, int64_t *b, int64_t *c, int64_t n,
                                                   int64_t m) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i * n) + j < n * m) {
        c[(i * n) + j] = a[i] / b[(i * n) + j];
    }
}

__global__ void negate_tensor_fp32_kernel(float *a, float *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        result[i] = -a[i];
    }
}

__global__ void negate_tensor_fp64_kernel(double *a, double *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        result[i] = -a[i];
    }
}

__global__ void negate_tensor_i8_kernel(int8_t *a, int8_t *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        result[i] = -a[i];
    }
}

__global__ void negate_tensor_i16_kernel(int16_t *a, int16_t *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        result[i] = -a[i];
    }
}

__global__ void negate_tensor_i32_kernel(int32_t *a, int32_t *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        result[i] = -a[i];
    }
}

__global__ void negate_tensor_i64_kernel(int64_t *a, int64_t *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        result[i] = -a[i];
    }
}

__global__ void abs_tensor_fp32_kernel(float *a, float *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        result[i] = fabsf(a[i]);
    }
}

__global__ void abs_tensor_fp64_kernel(double *a, double *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        result[i] = fabs(a[i]);
    }
}

__global__ void abs_tensor_i8_kernel(int8_t *a, int8_t *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int mask = a[i] >> (sizeof(int8_t) * CHAR_BIT - 1);
        result[i] = (a[i] + mask) ^ mask;
    }
}

__global__ void abs_tensor_i16_kernel(int16_t *a, int16_t *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int mask = a[i] >> (sizeof(int16_t) * CHAR_BIT - 1);
        result[i] = (a[i] + mask) ^ mask;
    }
}

__global__ void abs_tensor_i32_kernel(int32_t *a, int32_t *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int mask = a[i] >> (sizeof(int32_t) * CHAR_BIT - 1);
        result[i] = (a[i] + mask) ^ mask;
    }
}

__global__ void abs_tensor_i64_kernel(int64_t *a, int64_t *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int mask = a[i] >> (sizeof(int64_t) * CHAR_BIT - 1);
        result[i] = (a[i] + mask) ^ mask;
    }
}

__global__ void sqrt_tensor_fp32_kernel(float *a, float *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        result[i] = sqrtf(a[i]);
    }
}

__global__ void sqrt_tensor_fp64_kernel(double *a, double *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        result[i] = sqrt(a[i]);
    }
}

__global__ void sqrt_tensor_u8_kernel(uint8_t *a, uint8_t *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = static_cast<int8_t>(sqrtf(a[i]));
    }
}

__global__ void sqrt_tensor_i8_kernel(int8_t *a, int8_t *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = static_cast<int8_t>(sqrtf(a[i]));
    }
}

__global__ void sqrt_tensor_u16_kernel(uint16_t *a, uint16_t *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = static_cast<int16_t>(sqrtf(a[i]));
    }
}

__global__ void sqrt_tensor_i16_kernel(int16_t *a, int16_t *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = static_cast<int16_t>(sqrtf(a[i]));
    }
}

__global__ void sqrt_tensor_u32_kernel(uint32_t *a, uint32_t *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = static_cast<int32_t>(sqrtf(a[i]));
    }
}

__global__ void sqrt_tensor_i32_kernel(int32_t *a, int32_t *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = static_cast<int32_t>(sqrtf(a[i]));
    }
}

__global__ void sqrt_tensor_u64_kernel(uint64_t *a, uint64_t *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = static_cast<int64_t>(sqrtf(a[i]));
    }
}

__global__ void sqrt_tensor_i64_kernel(int64_t *a, int64_t *result, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = static_cast<int64_t>(sqrtf(a[i]));
    }
}

bool add_tensor_tensor_bf16_invoke(nv_bfloat16 *a, nv_bfloat16 *b, nv_bfloat16 *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    add_tensor_tensor_bf16_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_tensor_fp32_invoke(float *a, float *b, float *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    add_tensor_tensor_fp32_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_tensor_fp64_invoke(double *a, double *b, double *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    add_tensor_tensor_fp64_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    add_tensor_tensor_u8_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    add_tensor_tensor_i8_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    add_tensor_tensor_u16_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    add_tensor_tensor_i16_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    add_tensor_tensor_u32_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    add_tensor_tensor_i32_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    add_tensor_tensor_u64_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    add_tensor_tensor_i64_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_broadcast_tensor_fp32_invoke(float *a, float *b, float *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    add_tensor_broadcast_tensor_fp32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_broadcast_tensor_fp64_invoke(double *a, double *b, double *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    add_tensor_broadcast_tensor_fp64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_broadcast_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    add_tensor_broadcast_tensor_u8_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_broadcast_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    add_tensor_broadcast_tensor_i8_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_broadcast_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    add_tensor_broadcast_tensor_u16_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_broadcast_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    add_tensor_broadcast_tensor_i16_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_broadcast_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    add_tensor_broadcast_tensor_u32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_broadcast_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    add_tensor_broadcast_tensor_i32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_broadcast_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    add_tensor_broadcast_tensor_u64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_tensor_broadcast_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    add_tensor_broadcast_tensor_i64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_broadcast_tensor_tensor_fp32_invoke(float *a, float *b, float *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    add_broadcast_tensor_tensor_fp32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_broadcast_tensor_tensor_fp64_invoke(double *a, double *b, double *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    add_broadcast_tensor_tensor_fp64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_broadcast_tensor_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    add_broadcast_tensor_tensor_u8_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_broadcast_tensor_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    add_broadcast_tensor_tensor_i8_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_broadcast_tensor_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    add_broadcast_tensor_tensor_u16_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_broadcast_tensor_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    add_broadcast_tensor_tensor_i16_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_broadcast_tensor_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    add_broadcast_tensor_tensor_u32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_broadcast_tensor_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    add_broadcast_tensor_tensor_i32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_broadcast_tensor_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    add_broadcast_tensor_tensor_u64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool add_broadcast_tensor_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    add_broadcast_tensor_tensor_i64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_tensor_fp32_invoke(float *a, float *b, float *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    subtract_tensor_tensor_fp32_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_tensor_fp64_invoke(double *a, double *b, double *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    subtract_tensor_tensor_fp64_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    subtract_tensor_tensor_u8_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    subtract_tensor_tensor_i8_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    subtract_tensor_tensor_u16_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    subtract_tensor_tensor_i16_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    subtract_tensor_tensor_u32_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    subtract_tensor_tensor_i32_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    subtract_tensor_tensor_u64_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    subtract_tensor_tensor_i64_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_broadcast_tensor_fp32_invoke(float *a, float *b, float *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    subtract_tensor_broadcast_tensor_fp32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_broadcast_tensor_fp64_invoke(double *a, double *b, double *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    subtract_tensor_broadcast_tensor_fp64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_broadcast_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    subtract_tensor_broadcast_tensor_u8_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_broadcast_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    subtract_tensor_broadcast_tensor_i8_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_broadcast_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    subtract_tensor_broadcast_tensor_u16_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_broadcast_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    subtract_tensor_broadcast_tensor_i16_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_broadcast_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    subtract_tensor_broadcast_tensor_u32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_broadcast_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    subtract_tensor_broadcast_tensor_i32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_broadcast_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    subtract_tensor_broadcast_tensor_u64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_tensor_broadcast_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    subtract_tensor_broadcast_tensor_i64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_broadcast_tensor_tensor_fp32_invoke(float *a, float *b, float *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    subtract_broadcast_tensor_tensor_fp32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_broadcast_tensor_tensor_fp64_invoke(double *a, double *b, double *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    subtract_broadcast_tensor_tensor_fp64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_broadcast_tensor_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    subtract_broadcast_tensor_tensor_u8_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_broadcast_tensor_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    subtract_broadcast_tensor_tensor_i8_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_broadcast_tensor_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    subtract_broadcast_tensor_tensor_u16_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_broadcast_tensor_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    subtract_broadcast_tensor_tensor_i16_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_broadcast_tensor_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    subtract_broadcast_tensor_tensor_u32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_broadcast_tensor_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    subtract_broadcast_tensor_tensor_i32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_broadcast_tensor_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    subtract_broadcast_tensor_tensor_u64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool subtract_broadcast_tensor_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    subtract_broadcast_tensor_tensor_i64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_tensor_fp32_invoke(float *a, float *b, float *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    multiply_tensor_tensor_fp32_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_tensor_fp64_invoke(double *a, double *b, double *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    multiply_tensor_tensor_fp64_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    multiply_tensor_tensor_u8_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    multiply_tensor_tensor_i8_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    multiply_tensor_tensor_u16_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    multiply_tensor_tensor_i16_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    multiply_tensor_tensor_u32_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    multiply_tensor_tensor_i32_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    multiply_tensor_tensor_u64_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    multiply_tensor_tensor_i64_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_broadcast_tensor_fp32_invoke(float *a, float *b, float *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    multiply_tensor_broadcast_tensor_fp32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_broadcast_tensor_fp64_invoke(double *a, double *b, double *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);
    multiply_tensor_broadcast_tensor_fp64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_broadcast_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((n + block_size.x - 1) / block_size.x),
                   ((m + block_size.y - 1) / block_size.y));
    multiply_tensor_broadcast_tensor_u8_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_broadcast_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((n + block_size.x - 1) / block_size.x),
                   ((m + block_size.y - 1) / block_size.y));
    multiply_tensor_broadcast_tensor_i8_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_broadcast_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((n + block_size.x - 1) / block_size.x),
                   ((m + block_size.y - 1) / block_size.y));
    multiply_tensor_broadcast_tensor_u16_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_broadcast_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((n + block_size.x - 1) / block_size.x),
                   ((m + block_size.y - 1) / block_size.y));
    multiply_tensor_broadcast_tensor_i16_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_broadcast_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((n + block_size.x - 1) / block_size.x),
                   ((m + block_size.y - 1) / block_size.y));
    multiply_tensor_broadcast_tensor_u32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_broadcast_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((n + block_size.x - 1) / block_size.x),
                   ((m + block_size.y - 1) / block_size.y));
    multiply_tensor_broadcast_tensor_i32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_broadcast_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((n + block_size.x - 1) / block_size.x),
                   ((m + block_size.y - 1) / block_size.y));
    multiply_tensor_broadcast_tensor_u64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_tensor_broadcast_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((n + block_size.x - 1) / block_size.x),
                   ((m + block_size.y - 1) / block_size.y));
    multiply_tensor_broadcast_tensor_i64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_broadcast_tensor_tensor_fp32_invoke(float *a, float *b, float *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((m + block_size.x - 1) / block_size.x),
                   ((n + block_size.y - 1) / block_size.y));
    multiply_broadcast_tensor_tensor_fp32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_broadcast_tensor_tensor_fp64_invoke(double *a, double *b, double *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((m + block_size.x - 1) / block_size.x),
                   ((n + block_size.y - 1) / block_size.y));
    multiply_broadcast_tensor_tensor_fp64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_broadcast_tensor_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((m + block_size.x - 1) / block_size.x),
                   ((n + block_size.y - 1) / block_size.y));
    multiply_broadcast_tensor_tensor_u8_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_broadcast_tensor_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((m + block_size.x - 1) / block_size.x),
                   ((n + block_size.y - 1) / block_size.y));
    multiply_broadcast_tensor_tensor_i8_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_broadcast_tensor_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((m + block_size.x - 1) / block_size.x),
                   ((n + block_size.y - 1) / block_size.y));
    multiply_broadcast_tensor_tensor_u16_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_broadcast_tensor_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((m + block_size.x - 1) / block_size.x),
                   ((n + block_size.y - 1) / block_size.y));
    multiply_broadcast_tensor_tensor_i16_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_broadcast_tensor_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((m + block_size.x - 1) / block_size.x),
                   ((n + block_size.y - 1) / block_size.y));
    multiply_broadcast_tensor_tensor_u32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_broadcast_tensor_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((m + block_size.x - 1) / block_size.x),
                   ((n + block_size.y - 1) / block_size.y));
    multiply_broadcast_tensor_tensor_i32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_broadcast_tensor_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((m + block_size.x - 1) / block_size.x),
                   ((n + block_size.y - 1) / block_size.y));
    multiply_broadcast_tensor_tensor_u64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool multiply_broadcast_tensor_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((m + block_size.x - 1) / block_size.x),
                   ((n + block_size.y - 1) / block_size.y));
    multiply_broadcast_tensor_tensor_i64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_tensor_fp32_invoke(float *a, float *b, float *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    divide_tensor_tensor_fp32_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_tensor_fp64_invoke(double *a, double *b, double *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    divide_tensor_tensor_fp64_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    divide_tensor_tensor_u8_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    divide_tensor_tensor_i8_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    divide_tensor_tensor_u16_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    divide_tensor_tensor_i16_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    divide_tensor_tensor_u32_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    divide_tensor_tensor_i32_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    divide_tensor_tensor_u64_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *c, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    divide_tensor_tensor_i64_kernel<<<grid_size, block_size>>>(a, b, c, n);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_broadcast_tensor_fp32_invoke(float *a, float *b, float *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((n + block_size.x - 1) / block_size.x),
                   ((m + block_size.y - 1) / block_size.y));
    divide_tensor_broadcast_tensor_fp32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_broadcast_tensor_fp64_invoke(double *a, double *b, double *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((n + block_size.x - 1) / block_size.x),
                   ((m + block_size.y - 1) / block_size.y));
    divide_tensor_broadcast_tensor_fp64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_broadcast_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((n + block_size.x - 1) / block_size.x),
                   ((m + block_size.y - 1) / block_size.y));

    divide_tensor_broadcast_tensor_u8_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_broadcast_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((n + block_size.x - 1) / block_size.x),
                   ((m + block_size.y - 1) / block_size.y));

    divide_tensor_broadcast_tensor_i8_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_broadcast_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((n + block_size.x - 1) / block_size.x),
                   ((m + block_size.y - 1) / block_size.y));

    divide_tensor_broadcast_tensor_u16_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_broadcast_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(((n + block_size.x - 1) / block_size.x),
                   ((m + block_size.y - 1) / block_size.y));

    divide_tensor_broadcast_tensor_i16_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_broadcast_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);

    divide_tensor_broadcast_tensor_u32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_broadcast_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);

    divide_tensor_broadcast_tensor_i32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_broadcast_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);

    divide_tensor_broadcast_tensor_u64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_tensor_broadcast_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (m + block_size.y - 1) / block_size.y);

    divide_tensor_broadcast_tensor_i64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_broadcast_tensor_tensor_fp32_invoke(float *a, float *b, float *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);

    divide_broadcast_tensor_tensor_fp32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_broadcast_tensor_tensor_fp64_invoke(double *a, double *b, double *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);

    divide_broadcast_tensor_tensor_fp64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_broadcast_tensor_tensor_u8_invoke(uint8_t *a, uint8_t *b, uint8_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);

    divide_broadcast_tensor_tensor_u8_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_broadcast_tensor_tensor_i8_invoke(int8_t *a, int8_t *b, int8_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);

    divide_broadcast_tensor_tensor_i8_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_broadcast_tensor_tensor_u16_invoke(uint16_t *a, uint16_t *b, uint16_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);

    divide_broadcast_tensor_tensor_u16_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_broadcast_tensor_tensor_i16_invoke(int16_t *a, int16_t *b, int16_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);

    divide_broadcast_tensor_tensor_i16_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_broadcast_tensor_tensor_u32_invoke(uint32_t *a, uint32_t *b, uint32_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);

    divide_broadcast_tensor_tensor_u32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_broadcast_tensor_tensor_i32_invoke(int32_t *a, int32_t *b, int32_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);

    divide_broadcast_tensor_tensor_i32_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_broadcast_tensor_tensor_u64_invoke(uint64_t *a, uint64_t *b, uint64_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);

    divide_broadcast_tensor_tensor_u64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool divide_broadcast_tensor_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t *c, int64_t n, int64_t m) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    dim3 grid_size((m + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);

    divide_broadcast_tensor_tensor_i64_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
    return cudaGetLastError() == cudaSuccess;
}

bool negate_tensor_fp32_invoke(float *a, float *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    negate_tensor_fp32_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool negate_tensor_fp64_invoke(double *a, double *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    negate_tensor_fp64_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool negate_tensor_i8_invoke(int8_t *a, int8_t *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    negate_tensor_i8_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool negate_tensor_i16_invoke(int16_t *a, int16_t *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    negate_tensor_i16_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool negate_tensor_i32_invoke(int32_t *a, int32_t *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    negate_tensor_i32_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool negate_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    negate_tensor_i64_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool abs_tensor_fp32_invoke(float *a, float *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    abs_tensor_fp32_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool abs_tensor_fp64_invoke(double *a, double *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    abs_tensor_fp64_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool abs_tensor_i8_invoke(int8_t *a, int8_t *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    abs_tensor_i8_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool abs_tensor_i16_invoke(int16_t *a, int16_t *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    abs_tensor_i16_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool abs_tensor_i32_invoke(int32_t *a, int32_t *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    abs_tensor_i32_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool abs_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    abs_tensor_i64_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool sqrt_tensor_fp32_invoke(float *a, float *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    sqrt_tensor_fp32_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool sqrt_tensor_fp64_invoke(double *a, double *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    sqrt_tensor_fp64_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool sqrt_tensor_u8_invoke(uint8_t *a, uint8_t *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    sqrt_tensor_u8_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool sqrt_tensor_i8_invoke(int8_t *a, int8_t *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    sqrt_tensor_i8_kernel<<<grid_size, block_size>>>(a, b, n);
    
    return cudaGetLastError() == cudaSuccess;
}

bool sqrt_tensor_u16_invoke(uint16_t *a, uint16_t *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    sqrt_tensor_u16_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool sqrt_tensor_i16_invoke(int16_t *a, int16_t *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    sqrt_tensor_i16_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool sqrt_tensor_u32_invoke(uint32_t *a, uint32_t *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    sqrt_tensor_u32_kernel<<<grid_size, block_size>>>(a, b, n);
    
    return cudaGetLastError() == cudaSuccess;
}

bool sqrt_tensor_i32_invoke(int32_t *a, int32_t *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    sqrt_tensor_i32_kernel<<<grid_size, block_size>>>(a, b, n);
    return cudaGetLastError() == cudaSuccess;
}

bool sqrt_tensor_u64_invoke(uint64_t *a, uint64_t *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    sqrt_tensor_u64_kernel<<<grid_size, block_size>>>(a, b, n);
    
    return cudaGetLastError() == cudaSuccess;
}

bool sqrt_tensor_i64_invoke(int64_t *a, int64_t *b, int64_t n) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    sqrt_tensor_i64_kernel<<<grid_size, block_size>>>(a, b, n);
    
    return cudaGetLastError() == cudaSuccess;
}