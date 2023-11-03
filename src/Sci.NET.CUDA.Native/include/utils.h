//
// Created by reece on 01/08/2023.
//

#ifndef SCI_NET_NATIVE_UTILS_H
#define SCI_NET_NATIVE_UTILS_H


#include <cublas_v2.h>
#include "api.h"
#include "cuda_runtime_api.h"

inline sdnApiStatusCode guard_cuda_status(cudaError_t status) {
    if (status != cudaSuccess) {
        return sdnApiStatusCode::sdnInternalError;
    }

    return sdnApiStatusCode::sdnSuccess;
}

inline sdnApiStatusCode guard_cublas_status(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        return sdnApiStatusCode::sdnInternalError;
    }

    return sdnApiStatusCode::sdnSuccess;
}

#endif //SCI_NET_NATIVE_UTILS_H
