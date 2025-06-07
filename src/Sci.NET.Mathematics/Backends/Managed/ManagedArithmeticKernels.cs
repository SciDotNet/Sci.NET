// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Numerics;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedArithmeticKernels : IArithmeticKernels
{
    public void Negate<TNumber>(
        IMemoryBlock<TNumber> tensor,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        _ = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = -tensorBlock[i]);
    }

    public void Abs<TNumber>(
        IMemoryBlock<TNumber> tensor,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        _ = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Abs(tensorBlock[i]));
    }

    public void AbsGradient<TNumber>(IMemoryBlock<TNumber> tensor, IMemoryBlock<TNumber> gradient, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor;
        var gradientBlock = (SystemMemoryBlock<TNumber>)gradient;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        var i = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            1,
            i =>
            {
                var tensorValue = tensorBlock[i];
                var gradientValue = gradientBlock[i];

                if (tensorValue > TNumber.Zero)
                {
                    resultBlock[i] = gradientValue;
                }
#pragma warning disable IDE0045 // Conflicting warnings
                else if (tensorValue < TNumber.Zero)
#pragma warning restore IDE0045
                {
                    resultBlock[i] = TNumber.Zero - gradientValue;
                }
                else
                {
                    resultBlock[i] = TNumber.Zero;
                }
            });
    }

    public void AbsoluteDifference<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)left;
        var rightBlock = (SystemMemoryBlock<TNumber>)right;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        _ = LazyParallelExecutor.For(
            0,
            resultBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Abs(leftBlock[i] - rightBlock[i]));
    }

    public void Sqrt<TNumber>(
        IMemoryBlock<TNumber> tensor,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        _ = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = GenericMath.Sqrt(tensorBlock[i]));
    }
}