// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;
using Sci.NET.Accelerators.Attributes;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedArithmeticKernels : IArithmeticKernels
{
    public void AddTensorTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)left;
        var rightBlock = (SystemMemoryBlock<TNumber>)right;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = leftBlock[i] + rightBlock[i]);
    }

    public void AddTensorBroadcastTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)left;
        var rightBlock = (SystemMemoryBlock<TNumber>)right;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        LazyParallelExecutor.For(
            0,
            m,
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            (i, j) => resultBlock[(i * n) + j] = leftBlock[(i * n) + j] + rightBlock[j]);
    }

    public void AddBroadcastTensorTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)left;
        var rightBlock = (SystemMemoryBlock<TNumber>)right;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        LazyParallelExecutor.For(
            0,
            m,
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            (i, j) => resultBlock[(i * n) + j] = leftBlock[j] + rightBlock[(i * n) + j]);
    }

    public void SubtractTensorTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)left;
        var rightBlock = (SystemMemoryBlock<TNumber>)right;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = leftBlock[i] - rightBlock[i]);
    }

    public void SubtractTensorBroadcastTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)left;
        var rightBlock = (SystemMemoryBlock<TNumber>)right;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        LazyParallelExecutor.For(
            0,
            m,
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            (i, j) => resultBlock[(i * n) + j] = leftBlock[(i * n) + j] - rightBlock[j]);
    }

    public void SubtractBroadcastTensorTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)left;
        var rightBlock = (SystemMemoryBlock<TNumber>)right;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        LazyParallelExecutor.For(
            0,
            m,
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            (i, j) => resultBlock[(i * n) + j] = leftBlock[j] - rightBlock[(i * n) + j]);
    }

    public void MultiplyTensorTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)left;
        var rightBlock = (SystemMemoryBlock<TNumber>)right;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = leftBlock[i] * rightBlock[i]);
    }

    public void MultiplyTensorBroadcastTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)left;
        var rightBlock = (SystemMemoryBlock<TNumber>)right;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        LazyParallelExecutor.For(
            0,
            m,
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            (i, j) => resultBlock[(i * n) + j] = leftBlock[(i * n) + j] * rightBlock[j]);
    }

    public void MultiplyBroadcastTensorTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)left;
        var rightBlock = (SystemMemoryBlock<TNumber>)right;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        LazyParallelExecutor.For(
            0,
            m,
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            (i, j) => resultBlock[(i * n) + j] = leftBlock[j] * rightBlock[(i * n) + j]);
    }

    public void DivideTensorTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)left;
        var rightBlock = (SystemMemoryBlock<TNumber>)right;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = leftBlock[i] / rightBlock[i]);
    }

    public void DivideTensorBroadcastTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)left;
        var rightBlock = (SystemMemoryBlock<TNumber>)right;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        LazyParallelExecutor.For(
            0,
            m,
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            (i, j) => resultBlock[(i * n) + j] = leftBlock[(i * n) + j] / rightBlock[j]);
    }

    public void DivideBroadcastTensorTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)left;
        var rightBlock = (SystemMemoryBlock<TNumber>)right;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        LazyParallelExecutor.For(
            0,
            m,
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            (i, j) => resultBlock[(i * n) + j] = leftBlock[j] / rightBlock[(i * n) + j]);
    }

    public void Negate<TNumber>(
        IMemoryBlock<TNumber> tensor,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        LazyParallelExecutor.For(
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

        LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Abs(tensorBlock[i]));
    }

    public void Sqrt<TNumber>(
        IMemoryBlock<TNumber> tensor,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                resultBlock[i] = Unsafe.SizeOf<TNumber>() == 4
                    ? TNumber.CreateChecked(MathF.Sqrt(float.CreateChecked(tensorBlock[i])))
                    : TNumber.CreateChecked(Math.Sqrt(double.CreateChecked(tensorBlock[i])));
            });
    }

    [Kernel]
    public void SomeKernel<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, string result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)left;
        var rightBlock = (SystemMemoryBlock<TNumber>)right;

        var n = leftBlock.Length;

        LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => leftBlock[i] += rightBlock[i]);
    }
}