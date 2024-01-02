// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedPermutationKernels : IPermutationKernels
{
    public void Permute<TNumber>(ITensor<TNumber> source, ITensor<TNumber> result, int[] permutation)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var sourceBlock = (SystemMemoryBlock<TNumber>)source.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        LazyParallelExecutor.For(
            0,
            sourceBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sourceIndices = source.Shape.GetIndicesFromLinearIndex(i);

                var permutedIndices = new int[permutation.Length];

                for (var j = 0; j < permutation.Length; j++)
                {
                    permutedIndices[j] = sourceIndices[permutation[j]];
                }

                var permutedIndex = result.Shape.GetLinearIndex(permutedIndices);

                resultBlock[permutedIndex] = sourceBlock[i];
            });
    }
}