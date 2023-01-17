// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#include "omp.h"
#include "matmul.h"
#include "storage/memory_allocator.h"

void Sci::NET::BLAS::Native::CPU::Kernels::matmul_kernel_st_float32(const MemoryBlock* left,
                                                                    const MemoryBlock* right,
                                                                    MemoryBlock** result,
                                                                    const int64_t left_dim_x,
                                                                    const int64_t left_dim_y,
                                                                    const int64_t right_dim_x,
                                                                    const int64_t right_dim_y)
{
    const auto output = Storage::alloc(left_dim_x * right_dim_y, float32);

    const auto* left_ptr = static_cast<float*>(left->data());
    const auto* right_ptr = static_cast<float*>(right->data());
    auto* result_ptr = static_cast<float*>(output->data());

    for (int i = 0; i < left_dim_x; i++)
    {
        for (int j = 0; j < right_dim_y; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < left_dim_y; k++)
            {
                sum += left_ptr[(i * left_dim_y) + k] * right_ptr[(k * right_dim_y) + j];
            }
            result_ptr[(i + right_dim_y) + j] = sum;
        }
    }

    *result = output;
}

void Sci::NET::BLAS::Native::CPU::Kernels::matmul_kernel_st_float64(const MemoryBlock* left,
                                                                    const MemoryBlock* right,
                                                                    MemoryBlock** result,
                                                                    const int64_t left_dim_x,
                                                                    const int64_t left_dim_y,
                                                                    const int64_t right_dim_x,
                                                                    const int64_t right_dim_y)
{
    const auto output = Storage::alloc(left_dim_x * right_dim_y, float64);

    const auto* left_ptr = static_cast<double*>(left->data());
    const auto* right_ptr = static_cast<double*>(right->data());
    auto* result_ptr = static_cast<double*>(output->data());

    for (int i = 0; i < left_dim_x; i++)
    {
        for (int j = 0; j < right_dim_y; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < left_dim_y; k++)
            {
                sum += left_ptr[(i * left_dim_y) + k] * right_ptr[(k * right_dim_y) + j];
            }
            result_ptr[(i + right_dim_y) + j] = sum;
        }
    }

    *result = output;
}

void Sci::NET::BLAS::Native::CPU::Kernels::matmul_kernel_mt_float32(const MemoryBlock* left,
                                                                    const MemoryBlock* right,
                                                                    MemoryBlock** result,
                                                                    const int64_t left_dim_x,
                                                                    const int64_t left_dim_y,
                                                                    const int64_t right_dim_x,
                                                                    const int64_t right_dim_y)
{
    const auto output = Storage::alloc(left_dim_x * right_dim_y, float32);

    const auto* left_ptr = static_cast<float*>(left->data());
    const auto* right_ptr = static_cast<float*>(right->data());
    auto* result_ptr = static_cast<float*>(output->data());

#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < left_dim_x; i++)
    {
        for (int j = 0; j < right_dim_y; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < left_dim_y; k++)
            {
                sum += left_ptr[(i * left_dim_y) + k] * right_ptr[(k * right_dim_y) + j];
            }
            result_ptr[(i + right_dim_y) + j] = sum;
        }
    }

    *result = output;
}

void Sci::NET::BLAS::Native::CPU::Kernels::matmul_kernel_mt_float64(const MemoryBlock* left,
                                                                    const MemoryBlock* right,
                                                                    MemoryBlock** result,
                                                                    const int64_t left_dim_x,
                                                                    const int64_t left_dim_y,
                                                                    const int64_t right_dim_x,
                                                                    const int64_t right_dim_y)
{
    const auto output = Storage::alloc(left_dim_x * right_dim_y, float64);

    const auto* left_ptr = static_cast<double*>(left->data());
    const auto* right_ptr = static_cast<double*>(right->data());
    auto* result_ptr = static_cast<double*>(output->data());

#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < left_dim_x; i++)
    {
        for (int j = 0; j < right_dim_y; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < left_dim_y; k++)
            {
                sum += left_ptr[(i * left_dim_y) + k] * right_ptr[(k * right_dim_y) + j];
            }
            result_ptr[i + right_dim_y + j] = sum;
        }
    }

    *result = output;
}
