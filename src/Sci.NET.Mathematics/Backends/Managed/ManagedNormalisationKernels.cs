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

    public void BatchNorm1dForward<TNumber>(
        Matrix<TNumber> input,
        Tensors.Vector<TNumber> scale,
        Tensors.Vector<TNumber> bias,
        Tensors.Vector<TNumber> runningMean,
        Tensors.Vector<TNumber> runningVariance,
        Matrix<TNumber> result,
        Scalar<TNumber> epsilon)
        where TNumber : unmanaged, IRootFunctions<TNumber>, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)input.Memory;
        var resultMemory = (SystemMemoryBlock<TNumber>)result.Memory;
        var scaleMemory = (SystemMemoryBlock<TNumber>)scale.Memory;
        var biasMemory = (SystemMemoryBlock<TNumber>)bias.Memory;
        var runningMeanMemory = (SystemMemoryBlock<TNumber>)runningMean.Memory;
        var runningVarianceMemory = (SystemMemoryBlock<TNumber>)runningVariance.Memory;
        var epsilonValue = epsilon.Value;

        _ = LazyParallelExecutor.For(
            0,
            input.Rows,
            0,
            input.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) =>
            {
                var inputIndex = (i * input.Columns) + j;
                var resultIndex = (i * result.Columns) + j;

                var mean = runningMeanMemory[j];
                var variance = runningVarianceMemory[j] * runningVarianceMemory[i];
                var std = TNumber.Sqrt(variance + epsilonValue);
                var normalized = (inputMemory[inputIndex] - mean) / std;
                var scaled = normalized * scaleMemory[j];

                resultMemory[resultIndex] = scaled + biasMemory[j];
            });
    }

    public void ClipPrime<TNumber>(ITensor<TNumber> tensor, Tensor<TNumber> result, TNumber min, TNumber max)
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