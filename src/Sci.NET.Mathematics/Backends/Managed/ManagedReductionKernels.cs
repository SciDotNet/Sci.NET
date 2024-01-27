// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Numerics.Intrinsics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedReductionKernels : IReductionKernels
{
    public void ReduceAddAll<TNumber>(ITensor<TNumber> tensor, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorMemoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Memory;
        var partialSums = new ConcurrentDictionary<int, ISimdVector<TNumber>>();
        var vectorCount = SimdVector.Count<TNumber>();

        for (var index = 0; index < partialSums.Count; index++)
        {
            partialSums[index] = SimdVector.Create<TNumber>();
        }

        var done = 0L;

        if (tensor.Shape.ElementCount >= vectorCount)
        {
            done = LazyParallelExecutor.For(
                0,
                tensor.Shape.ElementCount,
                1,
                vectorCount,
                i =>
                {
                    var vector = tensorMemoryBlock.UnsafeGetVectorUnchecked<TNumber>(i);

                    _ = partialSums.AddOrUpdate(
                        Environment.CurrentManagedThreadId,
                        vector,
                        (_, sum) => sum.Add(vector));
                });
        }

        var finalSum = TNumber.Zero;

        for (var i = done * vectorCount; i < tensor.Shape.ElementCount; i++)
        {
            finalSum += tensorMemoryBlock[i];
        }

        foreach (var partialVectorSum in partialSums.Values)
        {
            finalSum += partialVectorSum.Sum();
        }

        resultMemoryBlock[0] = finalSum;
    }

    public void ReduceAddAxis<TNumber>(ITensor<TNumber> tensor, int[] axes, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorMemoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Memory;
        var tensorShape = tensor.Shape;
        var resultShape = result.Shape;

        _ = LazyParallelExecutor.For(
            0,
            result.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var resultIndices = resultShape.GetIndicesFromLinearIndex(i);
                var tensorIndices = new int[tensorShape.Rank];

                var resultIndex = 0;

                for (var tensorIndex = 0; tensorIndex < tensorShape.Rank; tensorIndex++)
                {
                    if (!axes.Contains(tensorIndex))
                    {
                        tensorIndices[tensorIndex] = resultIndices[resultIndex++];
                    }
                }

                var sum = TNumber.Zero;

                foreach (var index in GetAxisIndices(tensorShape, axes))
                {
                    for (var axisIndex = 0; axisIndex < axes.Length; axisIndex++)
                    {
                        tensorIndices[axes[axisIndex]] = index[axisIndex];
                    }

                    sum += tensorMemoryBlock[tensorShape.GetLinearIndex(tensorIndices)];
                }

                resultMemoryBlock[i] = sum;
            });
    }

    private static IEnumerable<int[]> GetAxisIndices(Shape shape, int[] axes)
    {
        var currentIndices = new int[axes.Length];
        var maxIndices = axes.Select(axis => shape[axis]).ToArray();
        return EnumerateIndices(currentIndices, maxIndices, 0);
    }

    private static IEnumerable<int[]> EnumerateIndices(int[] current, int[] max, int dim)
    {
        if (dim == current.Length)
        {
            yield return (int[])current.Clone();
        }
        else
        {
            for (current[dim] = 0; current[dim] < max[dim]; current[dim]++)
            {
                foreach (var indices in EnumerateIndices(current, max, dim + 1))
                {
                    yield return indices;
                }
            }
        }
    }
}