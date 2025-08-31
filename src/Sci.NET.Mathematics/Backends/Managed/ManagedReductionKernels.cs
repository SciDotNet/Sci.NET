// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Linq;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedReductionKernels : IReductionKernels
{
    public void ReduceAddAll<TNumber>(ITensor<TNumber> tensor, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorMemoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Memory;
        using var partialSums = new ThreadLocal<TNumber>(() => TNumber.Zero, true);

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var currentValue = tensorMemoryBlock[i];
                var partialSum = partialSums.Value;
                if (currentValue != TNumber.Zero)
                {
                    partialSums.Value = partialSum + currentValue;
                }
            });

        var finalSum = TNumber.Zero;

        foreach (var partialVectorSum in partialSums.Values)
        {
            finalSum += partialVectorSum;
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

    public void ReduceMeanAll<TNumber>(ITensor<TNumber> tensor, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorMemoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Memory;
        using var partialSums = new ThreadLocal<TNumber>(() => TNumber.Zero, true);

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var currentValue = tensorMemoryBlock[i];
                var partialSum = partialSums.Value;
                partialSums.Value = partialSum + currentValue;
            });

        var finalSum = TNumber.Zero;

        foreach (var partialVectorSum in partialSums.Values)
        {
            finalSum += partialVectorSum;
        }

        resultMemoryBlock[0] = finalSum / TNumber.CreateChecked(tensor.Shape.ElementCount);
    }

    public void ReduceMeanAxis<TNumber>(ITensor<TNumber> tensor, int[] axes, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorMemoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Memory;
        var tensorShape = tensor.Shape;
        var resultShape = result.Shape;
        var axesCount = (from axis in axes select tensorShape[axis]).Product();

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

                resultMemoryBlock[i] = sum / TNumber.CreateChecked(axesCount);
            });
    }

    public void ReduceMaxAll<TNumber>(ITensor<TNumber> tensor, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorMemoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Memory;
        using var partialMaximums = new ThreadLocal<TNumber>(GenericMath.MinValue<TNumber>, true);

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var currentValue = tensorMemoryBlock[i];
                var partialMax = partialMaximums.Value;
                if (currentValue > partialMax)
                {
                    partialMaximums.Value = currentValue;
                }
            });

        var max = TNumber.Zero;

        foreach (var partialVectorSum in partialMaximums.Values)
        {
            max = TNumber.Max(max, partialVectorSum);
        }

        resultMemoryBlock[0] = max;
    }

    public void ReduceMaxAxis<TNumber>(ITensor<TNumber> tensor, int[] axes, Tensor<TNumber> result)
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

                var max = tensorMemoryBlock[0];

                foreach (var index in GetAxisIndices(tensorShape, axes))
                {
                    for (var axisIndex = 0; axisIndex < axes.Length; axisIndex++)
                    {
                        tensorIndices[axes[axisIndex]] = index[axisIndex];
                    }

                    max = TNumber.Max(max, tensorMemoryBlock[tensorShape.GetLinearIndex(tensorIndices)]);
                }

                resultMemoryBlock[i] = max;
            });
    }

    public void ReduceMinAll<TNumber>(ITensor<TNumber> tensor, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorMemoryBlock = (SystemMemoryBlock<TNumber>)tensor.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Memory;
        using var partialMaximums = new ThreadLocal<TNumber>(GenericMath.MaxValue<TNumber>, true);

        _ = LazyParallelExecutor.For(
            0,
            tensor.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var currentValue = tensorMemoryBlock[i];
                var partialMax = partialMaximums.Value;
                if (currentValue < partialMax)
                {
                    partialMaximums.Value = currentValue;
                }
            });

        var min = GenericMath.MaxValue<TNumber>();

        foreach (var partialVectorSum in partialMaximums.Values)
        {
            min = TNumber.Min(min, partialVectorSum);
        }

        resultMemoryBlock[0] = min;
    }

    public void ReduceMinAxis<TNumber>(ITensor<TNumber> tensor, int[] axes, Tensor<TNumber> result)
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

                var min = tensorMemoryBlock[i];

                foreach (var index in GetAxisIndices(tensorShape, axes))
                {
                    for (var axisIndex = 0; axisIndex < axes.Length; axisIndex++)
                    {
                        tensorIndices[axes[axisIndex]] = index[axisIndex];
                    }

                    min = TNumber.Min(min, tensorMemoryBlock[tensorShape.GetLinearIndex(tensorIndices)]);
                }

                resultMemoryBlock[i] = min;
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