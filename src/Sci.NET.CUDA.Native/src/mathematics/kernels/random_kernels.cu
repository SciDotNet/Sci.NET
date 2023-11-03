//
// Created by reece on 02/08/2023.
//

#include <curand.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 256
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void setup_kernel(curandState *state, int seed, size_t n) {
    auto id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < n) {
        curand_init(seed, id, 0, &state[id]);
    }
}

__global__ void random_uniform_fp32_kernel(curandState *state, float *dst, float min, float range, size_t n) {
    auto id = threadIdx.x + blockIdx.x * blockDim.x;
    float x;

    if (id < n) {
        auto localState = state[id];
        x = curand_uniform(&localState);
        state[id] = localState;
        dst[id] = min + range * x;
    }
}

__global__ void random_uniform_fp64_kernel(curandState *state, double *dst, double min, double range, size_t n) {
    auto id = threadIdx.x + blockIdx.x * blockDim.x;
    double x;

    if (id < n) {
        auto localState = state[id];
        x = curand_uniform_double(&localState);
        state[id] = localState;
        dst[id] = min + range * x;
    }
}

__global__ void random_uniform_u8_kernel(curandState *state, uint8_t *dst, uint8_t min, uint8_t range, size_t n) {
    auto id = threadIdx.x + blockIdx.x * blockDim.x;
    double x;

    if (id < n) {
        auto localState = state[id];
        x = curand_uniform(&localState);
        state[id] = localState;
        dst[id] = (uint8_t) ((float) min + (float) range * x);
    }
}

__global__ void random_uniform_u16_kernel(curandState *state, uint16_t *dst, uint16_t min, uint16_t range, size_t n) {
    auto id = threadIdx.x + blockIdx.x * blockDim.x;
    double x;

    if (id < n) {
        auto localState = state[id];
        x = curand_uniform(&localState);
        state[id] = localState;
        dst[id] = (uint16_t) ((float) min + (float) range * x);
    }
}

__global__ void random_uniform_u32_kernel(curandState *state, uint32_t *dst, uint32_t min, uint32_t range, size_t n) {
    auto id = threadIdx.x + blockIdx.x * blockDim.x;
    double x;

    if (id < n) {
        auto localState = state[id];
        x = curand_uniform_double(&localState);
        state[id] = localState;
        dst[id] = (uint32_t) ((double) min + (double) range * x);
    }
}

__global__ void random_uniform_u64_kernel(curandState *state, uint64_t *dst, uint64_t min, uint64_t range, size_t n) {
    auto id = threadIdx.x + blockIdx.x * blockDim.x;
    double x;

    if (id < n) {
        auto localState = state[id];
        x = curand_uniform_double(&localState);
        state[id] = localState;
        dst[id] = (uint64_t) ((double) min + (double) range * x);
    }
}

__global__ void random_uniform_i8_kernel(curandState *state, int8_t *dst, int8_t min, int8_t range, size_t n) {
    auto id = threadIdx.x + blockIdx.x * blockDim.x;
    double x;

    if (id < n) {
        auto localState = state[id];
        x = curand_uniform(&localState);
        state[id] = localState;
        dst[id] = (int8_t) ((float) min + (float) range * x);
    }
}

__global__ void random_uniform_i16_kernel(curandState *state, int16_t *dst, int16_t min, int16_t range, size_t n) {
    auto id = threadIdx.x + blockIdx.x * blockDim.x;
    double x;

    if (id < n) {
        auto localState = state[id];
        x = curand_uniform(&localState);
        state[id] = localState;
        dst[id] = (int16_t) ((float) min + (float) range * x);
    }
}

__global__ void random_uniform_i32_kernel(curandState *state, int32_t *dst, int32_t min, int32_t range, size_t n) {
    auto id = threadIdx.x + blockIdx.x * blockDim.x;
    double x;

    if (id < n) {
        auto localState = state[id];
        x = curand_uniform_double(&localState);
        state[id] = localState;
        dst[id] = (int32_t) ((double) min + (double) range * x);
    }
}

__global__ void random_uniform_i64_kernel(curandState *state, int64_t *dst, int64_t min, int64_t range, size_t n) {
    auto id = threadIdx.x + blockIdx.x * blockDim.x;
    double x;

    if (id < n) {
        auto localState = state[id];
        x = curand_uniform(&localState);
        state[id] = localState;
        dst[id] = (int64_t) ((double) min + (double) range * x);
    }
}

bool random_uniform_fp64_invoke(double *dst,
                                double min,
                                double max,
                                size_t n,
                                long seed) {
    double range = max - min;

    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    curandState *devStates;

    cudaMalloc((void **) &devStates, n * sizeof(curandState));
    setup_kernel<<<grid_size, block_size>>>(devStates, seed, n);

    random_uniform_fp64_kernel<<<grid_size, block_size>>>(devStates, dst, min, range, n);
    cudaDeviceSynchronize();

    cudaFree(devStates);

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}

bool random_uniform_fp32_invoke(float *dst,
                                float min,
                                float max,
                                size_t n,
                                long seed) {
    float range = max - min;

    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    curandState *devStates;

    cudaMalloc((void **) &devStates, n * sizeof(curandState));
    setup_kernel<<<grid_size, block_size>>>(devStates, seed, n);

    random_uniform_fp32_kernel<<<grid_size, block_size>>>(devStates, dst, min, range, n);
    cudaDeviceSynchronize();

    cudaFree(devStates);

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}

bool random_uniform_u8_invoke(uint8_t *dst,
                              uint8_t min,
                              uint8_t max,
                              size_t n,
                              long seed) {
    uint8_t range = max - min;

    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    curandState *devStates;

    cudaMalloc((void **) &devStates, n * sizeof(curandState));
    setup_kernel<<<grid_size, block_size>>>(devStates, seed, n);

    random_uniform_u8_kernel<<<grid_size, block_size>>>(devStates, dst, min, range, n);
    cudaDeviceSynchronize();

    cudaFree(devStates);

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}

bool random_uniform_u16_invoke(uint16_t *dst,
                               uint16_t min,
                               uint16_t max,
                               size_t n,
                               long seed) {
    uint16_t range = max - min;

    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    curandState *devStates;

    cudaMalloc((void **) &devStates, n * sizeof(curandState));
    setup_kernel<<<grid_size, block_size>>>(devStates, seed, n);

    random_uniform_u16_kernel<<<grid_size, block_size>>>(devStates, dst, min, range, n);
    cudaDeviceSynchronize();

    cudaFree(devStates);

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}

bool random_uniform_u32_invoke(uint32_t *dst,
                               uint32_t min,
                               uint32_t max,
                               size_t n,
                               long seed) {
    uint32_t range = max - min;

    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    curandState *devStates;

    cudaMalloc((void **) &devStates, n * sizeof(curandState));
    setup_kernel<<<grid_size, block_size>>>(devStates, seed, n);

    random_uniform_u32_kernel<<<grid_size, block_size>>>(devStates, dst, min, range, n);
    cudaDeviceSynchronize();

    cudaFree(devStates);

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}

bool random_uniform_u64_invoke(uint64_t *dst,
                               uint64_t min,
                               uint64_t max,
                               size_t n,
                               long seed) {
    uint64_t range = max - min;

    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    curandState *devStates;

    cudaMalloc((void **) &devStates, n * sizeof(curandState));
    setup_kernel<<<grid_size, block_size>>>(devStates, seed, n);

    random_uniform_u64_kernel<<<grid_size, block_size>>>(devStates, dst, min, range, n);
    cudaDeviceSynchronize();

    cudaFree(devStates);

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}

bool random_uniform_i8_invoke(int8_t *dst,
                              int8_t min,
                              int8_t max,
                              size_t n,
                              long seed) {
    int8_t range = max - min;

    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    curandState *devStates;

    cudaMalloc((void **) &devStates, n * sizeof(curandState));
    setup_kernel<<<grid_size, block_size>>>(devStates, seed, n);

    random_uniform_i8_kernel<<<grid_size, block_size>>>(devStates, dst, min, range, n);
    cudaDeviceSynchronize();

    cudaFree(devStates);

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}

bool random_uniform_i16_invoke(int16_t *dst,
                               int16_t min,
                               int16_t max,
                               size_t n,
                               long seed) {
    int16_t range = max - min;

    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    curandState *devStates;

    cudaMalloc((void **) &devStates, n * sizeof(curandState));
    setup_kernel<<<grid_size, block_size>>>(devStates, seed, n);

    random_uniform_i16_kernel<<<grid_size, block_size>>>(devStates, dst, min, range, n);
    cudaDeviceSynchronize();

    cudaFree(devStates);

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}

bool random_uniform_i32_invoke(int32_t *dst,
                               int32_t min,
                               int32_t max,
                               size_t n,
                               long seed) {
    int32_t range = max - min;

    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    curandState *devStates;

    cudaMalloc((void **) &devStates, n * sizeof(curandState));
    setup_kernel<<<grid_size, block_size>>>(devStates, seed, n);

    random_uniform_i32_kernel<<<grid_size, block_size>>>(devStates, dst, min, range, n);
    cudaDeviceSynchronize();

    cudaFree(devStates);

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}

bool random_uniform_i64_invoke(int64_t *dst,
                               int64_t min,
                               int64_t max,
                               size_t n,
                               long seed) {
    int64_t range = max - min;

    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);

    curandState *devStates;

    cudaMalloc((void **) &devStates, n * sizeof(curandState));
    setup_kernel<<<grid_size, block_size>>>(devStates, seed, n);

    random_uniform_i64_kernel<<<grid_size, block_size>>>(devStates, dst, min, range, n);
    cudaDeviceSynchronize();

    cudaFree(devStates);

    cudaError_t error = cudaGetLastError();

    return error == cudaSuccess;
}