//
// Created by reece on 31/07/2023.
//

#include "cuda_memory_api.h"


sdnApiStatusCode allocate_memory(void **ptr, size_t size) {
    void* dataPtr;
    auto result = cudaMalloc(&dataPtr, size);
    *ptr = dataPtr;

    return guard_cuda_status(result);
}

sdnApiStatusCode free_memory(void *ptr) {
    auto result = cudaFree(ptr);
    return guard_cuda_status(result);
}

sdnApiStatusCode copy_memory_to_device(void *dst, void *src, size_t size) {
    auto result = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return guard_cuda_status(result);
}

sdnApiStatusCode copy_memory_to_host(void *dst, void *src, size_t size) {
    auto result = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return guard_cuda_status(result);
}

sdnApiStatusCode copy_memory_device_to_device(void *dst, void *src, size_t size) {
    auto result = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    return guard_cuda_status(result);
}
