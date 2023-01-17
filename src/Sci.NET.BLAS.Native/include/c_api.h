// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

// ReSharper disable CppInconsistentNaming
#ifndef SCI_NET_C_API_H
#define SCI_NET_C_API_H

#include <cstdint>

#include "SciDotNet/api.h"
#include "storage/memory_block.h"
#include "storage/number_types.h"

SDN_DLL_EXPORT_API allocate(int64_t count,
                            Sci::NET::BLAS::Native::number_type type,
                            Sci::NET::BLAS::Native::MemoryBlock** result);

SDN_DLL_EXPORT_API matmul(const Sci::NET::BLAS::Native::MemoryBlock* left,
                          const Sci::NET::BLAS::Native::MemoryBlock* right,
                          Sci::NET::BLAS::Native::MemoryBlock** result,
                          int64_t left_dim_x,
                          int64_t left_dim_y,
                          int64_t right_dim_x,
                          int64_t right_dim_y);

SDN_DLL_EXPORT_API setData(const Sci::NET::BLAS::Native::MemoryBlock* destination,
                           const void* source,
                           int64_t length);
SDN_DLL_EXPORT_API getData(const Sci::NET::BLAS::Native::MemoryBlock* source,
                           void* destination,
                           int64_t length);

#endif //SCI_NET_C_API_H
