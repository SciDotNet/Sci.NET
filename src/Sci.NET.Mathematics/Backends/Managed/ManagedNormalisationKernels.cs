// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedNormalisationKernels : INormalisationKernels
{
    public void Clip<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Min(TNumber.Max(tensorBlock[i], min), max));
    }

    public void ClipBackward<TNumber>(ITensor<TNumber> tensor, Tensor<TNumber> result, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = tensorBlock[i] > min && tensorBlock[i] < max ? TNumber.One : TNumber.Zero);
    }
}