// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedBroadcastingKernels : IBroadcastingKernels
{
    public void Broadcast<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result, long[] strides)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorHandle = (SystemMemoryBlock<TNumber>)tensor.Handle;
        var resultHandle = (SystemMemoryBlock<TNumber>)result.Handle;

        if (tensor.Shape.IsScalar)
        {
            LazyParallelExecutor.For(
                0,
                result.Shape.ElementCount,
                ManagedTensorBackend.ParallelizationThreshold,
                i => resultHandle[i] = tensorHandle[0]);
        }
        else
        {
            LazyParallelExecutor.For(
                0,
                result.Shape.ElementCount,
                ManagedTensorBackend.ParallelizationThreshold,
                i =>
                {
                    var indices = result.Shape.GetIndicesFromLinearIndex(i);
                    var sourceIndex = indices.Select((t, j) => t * strides[j]).Sum();

                    resultHandle[i] = tensorHandle[sourceIndex];
                });
        }
    }
}