// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#include <cstring>
#include "c_api.h"
#include "exception"
#include "kernels/matmul.h"
#include "storage/memory_allocator.h"

apiStatusCode_t allocate(const int64_t count,
                         const Sci::NET::BLAS::Native::number_type type,
                         Sci::NET::BLAS::Native::MemoryBlock** result)
{
    try
    {
        *result = Sci::NET::BLAS::Native::Storage::alloc(count, type);
        return sdnSuccess;
    }
    catch (const std::exception& e)
    {
        return sdnInvalidValue;
    }
}

apiStatusCode_t matmul(const Sci::NET::BLAS::Native::MemoryBlock* left,
                       const Sci::NET::BLAS::Native::MemoryBlock* right,
                       Sci::NET::BLAS::Native::MemoryBlock** result,
                       const int64_t left_dim_x,
                       const int64_t left_dim_y,
                       const int64_t right_dim_x,
                       const int64_t right_dim_y)
{
    if (left->type() != right->type())
    {
        return sdnInvalidValue;
    }
    switch (left->type())
    {
        case Sci::NET::BLAS::Native::float32:
            Sci::NET::BLAS::Native::CPU::Kernels::matmul_kernel_mt_float32(left,
                                                                           right,
                                                                           result,
                                                                           left_dim_x,
                                                                           left_dim_y,
                                                                           right_dim_x,
                                                                           right_dim_y);
            return sdnSuccess;
        case Sci::NET::BLAS::Native::float64:
            Sci::NET::BLAS::Native::CPU::Kernels::matmul_kernel_mt_float64(left,
                                                                           right,
                                                                           result,
                                                                           left_dim_x,
                                                                           left_dim_y,
                                                                           right_dim_x,
                                                                           right_dim_y);
            return sdnSuccess;
    }

    return sdnInvalidValue;
}

apiStatusCode_t setData(const Sci::NET::BLAS::Native::MemoryBlock* destination,
                        const void* source,
                        const int64_t length)
{
    if (length != destination->element_count())
    {
        return sdnInvalidValue;
    }
    try
    {
        memcpy(destination->data(), source, length);
        return sdnSuccess;
    }
    catch (std::exception& ex)
    {
        return sdnInternalError;
    }
}

apiStatusCode_t getData(const Sci::NET::BLAS::Native::MemoryBlock* source,
                        void* destination,
                        const int64_t length)
{
    if (source->element_count() != length)
    {
        return sdnInvalidValue;
    }
    try
    {
        memcpy(destination, source->data(), length);
        return sdnSuccess;
    }
    catch (std::exception& ex)
    {
        return sdnInternalError;
    }
}
