// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#include <cstring>
#include "c_api.h"
#include "exception"
#include "kernels/matmul.h"
#include "storage/memory_allocator.h"

apiStatusCode_t sdn_allocate(int64_t count,
                             number_type type,
                             Storage::MemoryBlock **result) {
    try {
        *result = Storage::alloc(count, type);
        return sdnSuccess;
    }
    catch (const std::exception &e) {
        return sdnInvalidValue;
    }
}

apiStatusCode_t sdn_matmul(const Storage::MemoryBlock *left,
                           const Storage::MemoryBlock *right,
                           Storage::MemoryBlock **result,
                           int64_t left_dim_x,
                           int64_t left_dim_y,
                           int64_t right_dim_x,
                           int64_t right_dim_y) {
    if (left->type() != right->type()) {
        return sdnInvalidValue;
    }
    switch (left->type()) {
        case number_type::float32:
            CPU::Kernels::matmul_kernel_mt_float32(left,
                                                   right,
                                                   result,
                                                   left_dim_x,
                                                   left_dim_y,
                                                   right_dim_x,
                                                   right_dim_y);
            return sdnSuccess;
        case number_type::float64:
            CPU::Kernels::matmul_kernel_mt_float64(left,
                                                   right,
                                                   result,
                                                   left_dim_x,
                                                   left_dim_y,
                                                   right_dim_x,
                                                   right_dim_y);
            return sdnSuccess;
        case number_type::uint8:
            break;
    }

    return sdnInvalidValue;
}

apiStatusCode_t sdn_set_data(const Storage::MemoryBlock *destination,
                             const void *source) {
    try {
        memcpy(destination->data(), source, destination->data_size());
        return sdnSuccess;
    }
    catch (std::exception &ex) {
        return sdnInternalError;
    }
}

apiStatusCode_t sdn_get_data(const Storage::MemoryBlock *source,
                             void *destination) {
    try {
        memcpy(destination, source->data(), source->data_size());
        return sdnSuccess;
    }
    catch (std::exception &ex) {
        return sdnInternalError;
    }
}

apiStatusCode_t sdn_free(Storage::MemoryBlock *memoryBlock) {
    try {
        std::free(memoryBlock->data());
        delete memoryBlock;
        return sdnSuccess;
    }
    catch (std::exception &ex) {
        return sdnInternalError;
    }
}
