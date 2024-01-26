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
        var tensorHandle = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultHandle = (SystemMemoryBlock<TNumber>)result.Memory;

        if (tensor.Shape.IsScalar)
        {
            _ = LazyParallelExecutor.For(
                0,
                result.Shape.ElementCount,
                ManagedTensorBackend.ParallelizationThreshold,
                i => resultHandle[i] = tensorHandle[0]);
        }
        else
        {
            for (var i = 0; i < result.Shape.ElementCount; i++)
            {
                var resultDims = result.Shape.GetIndicesFromLinearIndex(i);
                var sourceDataOffset = resultDims.Select((t, j) => t * strides[j]).Sum();

                resultHandle[i] = tensorHandle[sourceDataOffset];
            }
        }
    }
}