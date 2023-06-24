// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedLinqKernels : ILinqKernels
{
    public void Map<TTensor, TNumber>(
        ITensor<TNumber> tensor,
        ITensor<TNumber> result,
        Func<TNumber, TNumber> action)
        where TTensor : class, ITensor<TNumber>
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = action(tensorBlock[i]));
    }
}