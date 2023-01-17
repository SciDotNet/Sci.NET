// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#ifndef SCI_NET_NUMBER_TYPES_H
#define SCI_NET_NUMBER_TYPES_H

#include "SciDotNet/api.h"

namespace Sci::NET::BLAS::Native {
    enum number_type {
        float32 = 1,
        float64 = 2
    };

    size_t sizeof_number(number_type type);
}

#endif //SCI_NET_NUMBER_TYPES_H
