//
// Created by reece on 01/08/2023.
//

#include "cuda_settings.h"
#include "utils.h"

thread_local static cublasHandle_t cublas_handle;

sdnApiStatusCode get_cuda_device_props(int device_id, cudaDeviceProp *props) {
    return guard_cuda_status(cudaGetDeviceProperties(props, device_id));
}

sdnApiStatusCode get_cuda_device_count(int *count) {
    return guard_cuda_status(cudaGetDeviceCount(count));
}

sdnApiStatusCode set_cublas_tensor_core_mode(bool enabled) {
    return guard_cublas_status(cublasSetMathMode(get_cublas_handle(), enabled ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
}

inline cublasHandle_t get_cublas_handle() {
    if (cublas_handle == nullptr) {
        cublasCreate(&cublas_handle);
    }

    return cublas_handle;
}
