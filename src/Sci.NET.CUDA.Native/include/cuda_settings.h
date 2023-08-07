//
// Created by reece on 01/08/2023.
//

#ifndef SCI_NET_NATIVE_CUDA_SETTINGS_H
#define SCI_NET_NATIVE_CUDA_SETTINGS_H

#include <cublas_v2.h>
#include "api.h"
#include "cuda_runtime_api.h"

SDN_DLL_EXPORT_API get_cuda_device_props(int device_id, cudaDeviceProp *props);

SDN_DLL_EXPORT_API get_cuda_device_count(int *count);

SDN_DLL_EXPORT_API set_cublas_tensor_core_mode(bool enabled);

inline cublasHandle_t get_cublas_handle();

#endif //SCI_NET_NATIVE_CUDA_SETTINGS_H
