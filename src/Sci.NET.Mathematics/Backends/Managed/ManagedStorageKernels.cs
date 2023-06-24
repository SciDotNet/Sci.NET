// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedStorageKernels : ITensorStorageKernels
{
    public IMemoryBlock<TNumber> Allocate<TNumber>(Shape shape)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return new SystemMemoryBlock<TNumber>(shape.ElementCount);
    }
}