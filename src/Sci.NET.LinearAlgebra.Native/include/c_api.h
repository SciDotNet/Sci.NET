// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#ifndef SCI_NET_C_API_H
#define SCI_NET_C_API_H

#include <cstdint>

#include "SciDotNet/api.h"
#include "storage/memory_block.h"
#include "storage/number_types.h"

using namespace Sci::NET::LinearAlgebra::Native;

SDN_DLL_EXPORT_API sdn_allocate(int64_t count,
                                Sci::NET::LinearAlgebra::Native::number_type type,
                                Sci::NET::LinearAlgebra::Native::Storage::MemoryBlock **result);

SDN_DLL_EXPORT_API sdn_free(Sci::NET::LinearAlgebra::Native::Storage::MemoryBlock *memoryBlock);

SDN_DLL_EXPORT_API sdn_set_data(const Sci::NET::LinearAlgebra::Native::Storage::MemoryBlock *destination,
                                const void *source);

SDN_DLL_EXPORT_API sdn_get_data(const Sci::NET::LinearAlgebra::Native::Storage::MemoryBlock *source,
                                void *destination);

SDN_DLL_EXPORT_API sdn_matmul(const Sci::NET::LinearAlgebra::Native::Storage::MemoryBlock *left,
                              const Sci::NET::LinearAlgebra::Native::Storage::MemoryBlock *right,
                              Sci::NET::LinearAlgebra::Native::Storage::MemoryBlock **result,
                              int64_t left_dim_x,
                              int64_t left_dim_y,
                              int64_t right_dim_x,
                              int64_t right_dim_y);

#endif //SCI_NET_C_API_H
