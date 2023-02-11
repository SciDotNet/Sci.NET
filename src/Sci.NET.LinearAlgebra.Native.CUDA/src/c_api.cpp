// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#include <exception>
#include "cuda_runtime.h"
#include "c_api.h"

using namespace Sci::NET::LinearAlgebra::Native::Storage;

apiStatusCode_t sdn_allocate(int64_t count,
                             number_type type,
                             Storage::MemoryBlock **result) {
    try {
        void* ptr = nullptr;

        if (cudaMalloc(&ptr, count * sizeof_number(type)) != cudaSuccess) {
            return sdnInternalError;
        }
        
        *result = new MemoryBlock(count, ptr, type);        
        return sdnSuccess;
    }
    catch (const std::exception &e) {
        return sdnInvalidValue;
    }
}