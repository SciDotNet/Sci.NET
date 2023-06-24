// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedReductionKernels : IReductionKernels
{
    public void ReduceAddAll<TNumber>(ITensor<TNumber> tensor, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var elementCount = tensor.Shape.ElementCount;
        var tensorMemory = (SystemMemoryBlock<TNumber>)tensor.Handle;
        var maxDegreeOfParallelism = Environment.ProcessorCount;
        var partitionSize = (int)Math.Ceiling((double)elementCount / maxDegreeOfParallelism);
        var partialSums = new TNumber[maxDegreeOfParallelism];

        LazyParallelExecutor.For(
            0,
            maxDegreeOfParallelism,
            maxDegreeOfParallelism,
            i =>
            {
                var start = i * partitionSize;
                var end = Math.Min(start + partitionSize, (int)elementCount);
                var sum = default(TNumber);

                for (var j = start; j < end; j++)
                {
                    sum += tensorMemory[j];
                }

                partialSums[i] = sum;
            });

        var sum = TNumber.Zero;

        for (var i = 0; i < maxDegreeOfParallelism; i++)
        {
            sum += partialSums[i];
        }

        ((SystemMemoryBlock<TNumber>)result.Handle)[0] = sum;
    }

    public void ReduceAddAllKeepDims<TNumber>(ITensor<TNumber> tensor, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorMemoryBlock = (SystemMemoryBlock<TNumber>)tensor.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;
        var count = tensor.Shape.ElementCount;
        var tensorShape = tensor.Shape;
        var resultShape = result.Shape;

        var strides = new long[tensorShape.Rank];

        for (long i = tensorShape.Rank - 1, stride = 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= tensorShape.Dimensions[i];
        }

        for (var i = 0; i < count; i++)
        {
            var indices = resultShape.GetIndicesFromLinearIndex(i);
            var tensorIndex = GetTensorIndex(indices, strides);
            resultMemoryBlock[i] = tensorMemoryBlock[tensorIndex];
        }
    }

    public void ReduceAddAxis<TNumber>(ITensor<TNumber> tensor, int[] axes, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorMemoryBlock = (SystemMemoryBlock<TNumber>)tensor.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;
        var tensorShape = tensor.Shape;
        var resultShape = result.Shape;
        var maxDegreeOfParallelism = Environment.ProcessorCount;

        _ = Parallel.ForEach(
            axes,
            axis =>
            {
                var axisSize = tensorShape[axis];
                var partialSums = new TNumber[maxDegreeOfParallelism];

                for (var i = 0; i < maxDegreeOfParallelism; i++)
                {
                    partialSums[i] = TNumber.Zero;
                }

                var partitionSize = (int)Math.Ceiling((double)axisSize / maxDegreeOfParallelism);

                LazyParallelExecutor.For(
                    0,
                    maxDegreeOfParallelism,
                    maxDegreeOfParallelism,
                    i =>
                    {
                        var start = (int)i * partitionSize;
                        var end = Math.Min(start + partitionSize, axisSize);
                        var sum = TNumber.Zero;

                        for (var j = start; j < end; j++)
                        {
                            var tensorIndices = new int[tensorShape.Rank];
                            tensorIndices[axis] = j;

                            var tensorIndex = tensorShape.GetLinearIndex(tensorIndices);
                            sum += tensorMemoryBlock[tensorIndex];
                        }

                        partialSums[i] = sum;
                    });

                var totalSum = TNumber.Zero;

                for (var i = 0; i < maxDegreeOfParallelism; i++)
                {
                    totalSum += partialSums[i];
                }

                var resultIndices = new int[resultShape.Rank];
                resultIndices[axis] = 0;
                resultMemoryBlock[resultShape.GetLinearIndex(resultIndices)] = totalSum;
            });
    }

    public void ReduceAddAxisKeepDims<TNumber>(ITensor<TNumber> tensor, int[] axes, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorMemoryBlock = (SystemMemoryBlock<TNumber>)tensor.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;
        var tensorShape = tensor.Shape;
        var resultShape = result.Shape;

        var tensorIndices = new int[tensorShape.Rank];
        var resultIndices = new int[resultShape.Rank];

        _ = Parallel.ForEach(
            axes,
            axis =>
            {
                var axisSize = tensorShape[axis];

                for (var i = 0; i < tensorShape.Rank; i++)
                {
                    tensorIndices[i] = 0;
                }

                for (var i = 0; i < resultShape.Rank; i++)
                {
                    resultIndices[i] = 0;
                }

                for (var i = 0; i < axisSize; i++)
                {
                    tensorIndices[axis] = i;
                    resultIndices[axis] = 0;

                    var tensorIndex = tensorShape.GetLinearIndex(tensorIndices);
                    var resultIndex = resultShape.GetLinearIndex(resultIndices);

                    resultMemoryBlock[resultIndex] += tensorMemoryBlock[tensorIndex];
                }
            });
    }

    private static int GetTensorIndex(IEnumerable<int> indices, IReadOnlyList<long> strides)
    {
        var index = indices.Select((t, i) => strides[i] * t).Sum();

        return (int)index;
    }
}