// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedActivationFunctionKernels : IActivationFunctionKernels
{
    public void Sigmoid<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => outputMemory[i] = TNumber.One / (TNumber.One + TNumber.Exp(-inputMemory[i])));
    }

    public void SigmoidPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sigmoid = TNumber.One / (TNumber.One + TNumber.Exp(-inputMemory[i]));
                outputMemory[i] = sigmoid * (TNumber.One - sigmoid);
            });
    }

    public void ReLU<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => outputMemory[i] = inputMemory[i] > TNumber.Zero ? inputMemory[i] : TNumber.Zero);
    }

    public void ReLUPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => outputMemory[i] = inputMemory[i] > TNumber.Zero ? TNumber.One : TNumber.Zero);
    }
}