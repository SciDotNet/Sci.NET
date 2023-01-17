// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#ifndef SCI_NET_MEMORY_BLOCK_H
#define SCI_NET_MEMORY_BLOCK_H

#include <cstdint>
#include "number_types.h"
#include "SciDotNet/api.h"

namespace Sci::NET::BLAS::Native
{
    SDN_DLL_EXPORT_CLASS MemoryBlock
    {
    public:
        MemoryBlock(int64_t count, void* data, number_type number_type);
        [[nodiscard]] int64_t element_count() const;
        [[nodiscard]] number_type type() const;
        [[nodiscard]] void* data() const;

    private:
        int64_t element_count_;
        void* data_ptr_;
        number_type number_type_;
    };
}

#endif //SCI_NET_MEMORY_BLOCK_H
