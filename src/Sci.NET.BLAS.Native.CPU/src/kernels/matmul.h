// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#ifndef SCI_NET_MATMUL_H
#define SCI_NET_MATMUL_H

#include "storage/memory_block.h"

namespace Sci::NET::BLAS::Native::CPU::Kernels
{
    using namespace Native;

    void matmul_kernel_st_float32(const MemoryBlock* left,
                                  const MemoryBlock* right,
                                  MemoryBlock** result,
                                  int64_t left_dim_x,
                                  int64_t left_dim_y,
                                  int64_t right_dim_x,
                                  int64_t right_dim_y);

    void matmul_kernel_st_float64(const MemoryBlock* left,
                                  const MemoryBlock* right,
                                  MemoryBlock** result,
                                  int64_t left_dim_x,
                                  int64_t left_dim_y,
                                  int64_t right_dim_x,
                                  int64_t right_dim_y);

    void matmul_kernel_mt_float32(const MemoryBlock* left,
                                  const MemoryBlock* right,
                                  MemoryBlock** result,
                                  int64_t left_dim_x,
                                  int64_t left_dim_y,
                                  int64_t right_dim_x,
                                  int64_t right_dim_y);

    void matmul_kernel_mt_float64(const MemoryBlock* left,
                                  const MemoryBlock* right,
                                  MemoryBlock** result,
                                  int64_t left_dim_x,
                                  int64_t left_dim_y,
                                  int64_t right_dim_x,
                                  int64_t right_dim_y);
}

#endif //SCI_NET_MATMUL_H
