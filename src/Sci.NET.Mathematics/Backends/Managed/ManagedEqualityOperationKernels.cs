// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedEqualityOperationKernels : IEqualityOperationKernels
{
    public unsafe void PointwiseEqualsKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;
        var canVectorize = Vector.IsHardwareAccelerated && Avx.IsSupported;

        if (canVectorize && typeof(TNumber) == typeof(float))
        {
            PointwiseEqualsFloat32Kernel(
                (float*)leftBlock.Pointer,
                (float*)rightBlock.Pointer,
                (float*)resultBlock.Pointer,
                n);
            return;
        }

        if (canVectorize && typeof(TNumber) == typeof(double))
        {
            PointwiseEqualsFloat64Kernel(
                (double*)leftBlock.Pointer,
                (double*)rightBlock.Pointer,
                (double*)resultBlock.Pointer,
                n);
            return;
        }

        _ = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = leftBlock[i] == rightBlock[i] ? TNumber.One : TNumber.Zero);
    }

    public unsafe void PointwiseNotEqualKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;
        var canVectorize = Vector.IsHardwareAccelerated && Avx.IsSupported;

        if (canVectorize && typeof(TNumber) == typeof(float))
        {
            PointwiseNotEqualFloat32Kernel(
                (float*)leftBlock.Pointer,
                (float*)rightBlock.Pointer,
                (float*)resultBlock.Pointer,
                n);
            return;
        }

        if (canVectorize && typeof(TNumber) == typeof(double))
        {
            PointwiseNotEqualFloat64Kernel(
                (double*)leftBlock.Pointer,
                (double*)rightBlock.Pointer,
                (double*)resultBlock.Pointer,
                n);
            return;
        }

        _ = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = leftBlock[i] == rightBlock[i] ? TNumber.Zero : TNumber.One);
    }

    public unsafe void PointwiseGreaterThanKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;
        var canVectorize = Vector.IsHardwareAccelerated && Avx.IsSupported;

        if (canVectorize && typeof(TNumber) == typeof(float))
        {
            PointwiseGreaterThanFloat32Kernel(
                (float*)leftBlock.Pointer,
                (float*)rightBlock.Pointer,
                (float*)resultBlock.Pointer,
                n);
            return;
        }

        if (canVectorize && typeof(TNumber) == typeof(double))
        {
            PointwiseGreaterThanFloat64Kernel(
                (double*)leftBlock.Pointer,
                (double*)rightBlock.Pointer,
                (double*)resultBlock.Pointer,
                n);
            return;
        }

        _ = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = leftBlock[i] > rightBlock[i] ? TNumber.One : TNumber.Zero);
    }

    public unsafe void PointwiseGreaterThanOrEqualKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;
        var canVectorize = Vector.IsHardwareAccelerated && Avx.IsSupported;

        if (canVectorize && typeof(TNumber) == typeof(float))
        {
            PointwiseGreaterThanOrEqualFloat32Kernel(
                (float*)leftBlock.Pointer,
                (float*)rightBlock.Pointer,
                (float*)resultBlock.Pointer,
                n);
            return;
        }

        if (canVectorize && typeof(TNumber) == typeof(double))
        {
            PointwiseGreaterThanOrEqualFloat64Kernel(
                (double*)leftBlock.Pointer,
                (double*)rightBlock.Pointer,
                (double*)resultBlock.Pointer,
                n);
            return;
        }

        _ = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = leftBlock[i] >= rightBlock[i] ? TNumber.One : TNumber.Zero);
    }

    public void PointwiseLessThanKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        for (var i = 0; i < n; i++)
        {
            resultBlock[i] = leftBlock[i].CompareTo(rightBlock[i]) < 0 ? TNumber.One : TNumber.Zero;
        }
    }

    public void PointwiseLessThanOrEqualKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)leftOperand;
        var rightBlock = (SystemMemoryBlock<TNumber>)rightOperand;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;

        for (var i = 0; i < n; i++)
        {
            resultBlock[i] = leftBlock[i].CompareTo(rightBlock[i]) <= 0 ? TNumber.One : TNumber.Zero;
        }
    }

    private static unsafe void PointwiseEqualsFloat32Kernel(float* leftBlockPointer, float* rightBlockPointer, float* resultBlockPointer, long n)
    {
        var vectorSize = Vector256<float>.Count;
        var one = Vector256<float>.One;

        var done = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorSize,
            i =>
            {
                var leftVector = Avx.LoadVector256(leftBlockPointer + i);
                var rightVector = Avx.LoadVector256(rightBlockPointer + i);
                var equals = Avx.CompareEqual(leftVector, rightVector);
                var res = Avx.And(equals, one);
                res.Store(resultBlockPointer + i);
            });

        for (var i = done; i < n; i++)
        {
            resultBlockPointer[i] = leftBlockPointer[i].Equals(rightBlockPointer[i]) ? 1f : 0f;
        }
    }

    private static unsafe void PointwiseEqualsFloat64Kernel(double* leftBlockPointer, double* rightBlockPointer, double* resultBlockPointer, long n)
    {
        var vectorSize = Vector256<double>.Count;
        var one = Vector256<double>.One;

        var done = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorSize,
            i =>
            {
                var leftVector = Avx.LoadVector256(leftBlockPointer + i);
                var rightVector = Avx.LoadVector256(rightBlockPointer + i);
                var equals = Avx.CompareEqual(leftVector, rightVector);
                var res = Avx.And(equals, one);
                res.Store(resultBlockPointer + i);
            });

        for (var i = done; i < n; i++)
        {
            resultBlockPointer[i] = leftBlockPointer[i].Equals(rightBlockPointer[i]) ? 1f : 0f;
        }
    }

    private static unsafe void PointwiseNotEqualFloat32Kernel(float* leftBlockPointer, float* rightBlockPointer, float* resultBlockPointer, long n)
    {
        var vectorSize = Vector256<float>.Count;
        var one = Vector256<float>.One;

        var done = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorSize,
            i =>
            {
                var leftVector = Avx.LoadVector256(leftBlockPointer + i);
                var rightVector = Avx.LoadVector256(rightBlockPointer + i);
                var equals = Avx.CompareEqual(leftVector, rightVector);
                var res = Avx.AndNot(equals, one);
                res.Store(resultBlockPointer + i);
            });

        for (var i = done; i < n; i++)
        {
            resultBlockPointer[i] = leftBlockPointer[i].Equals(rightBlockPointer[i]) ? 0f : 1f;
        }
    }

    private static unsafe void PointwiseNotEqualFloat64Kernel(double* leftBlockPointer, double* rightBlockPointer, double* resultBlockPointer, long n)
    {
        var vectorSize = Vector256<double>.Count;
        var one = Vector256<double>.One;

        var done = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorSize,
            i =>
            {
                var leftVector = Avx.LoadVector256(leftBlockPointer + i);
                var rightVector = Avx.LoadVector256(rightBlockPointer + i);
                var equals = Avx.CompareEqual(leftVector, rightVector);
                var res = Avx.AndNot(equals, one);
                res.Store(resultBlockPointer + i);
            });

        for (var i = done; i < n; i++)
        {
            resultBlockPointer[i] = leftBlockPointer[i].Equals(rightBlockPointer[i]) ? 0f : 1f;
        }
    }

    private static unsafe void PointwiseGreaterThanFloat32Kernel(float* leftBlockPointer, float* rightBlockPointer, float* resultBlockPointer, long n)
    {
        var vectorSize = Vector256<float>.Count;

        var done = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorSize,
            i =>
            {
                var leftVector = Avx.LoadVector256(leftBlockPointer + i);
                var rightVector = Avx.LoadVector256(rightBlockPointer + i);
                var greaterThan = Avx.CompareGreaterThan(leftVector, rightVector);
                greaterThan.Store(resultBlockPointer + i);
            });

        for (var i = done; i < n; i++)
        {
            resultBlockPointer[i] = leftBlockPointer[i] > rightBlockPointer[i] ? 1f : 0f;
        }
    }

    private static unsafe void PointwiseGreaterThanFloat64Kernel(double* leftBlockPointer, double* rightBlockPointer, double* resultBlockPointer, long n)
    {
        var vectorSize = Vector256<double>.Count;

        var done = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorSize,
            i =>
            {
                var leftVector = Avx.LoadVector256(leftBlockPointer + i);
                var rightVector = Avx.LoadVector256(rightBlockPointer + i);
                var greaterThan = Avx.CompareGreaterThan(leftVector, rightVector);
                greaterThan.Store(resultBlockPointer + i);
            });

        for (var i = done; i < n; i++)
        {
            resultBlockPointer[i] = leftBlockPointer[i] > rightBlockPointer[i] ? 1d : 0d;
        }
    }

    private static unsafe void PointwiseGreaterThanOrEqualFloat32Kernel(float* leftBlockPointer, float* rightBlockPointer, float* resultBlockPointer, long n)
    {
        var vectorSize = Vector256<float>.Count;

        var done = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorSize,
            i =>
            {
                var leftVector = Avx.LoadVector256(leftBlockPointer + i);
                var rightVector = Avx.LoadVector256(rightBlockPointer + i);
                var greaterThanOrEqual = Avx.CompareGreaterThanOrEqual(leftVector, rightVector);
                greaterThanOrEqual.Store(resultBlockPointer + i);
            });

        for (var i = done; i < n; i++)
        {
            resultBlockPointer[i] = leftBlockPointer[i] >= rightBlockPointer[i] ? 1f : 0f;
        }
    }

    private static unsafe void PointwiseGreaterThanOrEqualFloat64Kernel(double* leftBlockPointer, double* rightBlockPointer, double* resultBlockPointer, long n)
    {
        var vectorSize = Vector256<double>.Count;
        var zeros = Vector256<double>.Zero;
        var ones = Vector256<double>.One;

        var done = LazyParallelExecutor.For(
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorSize,
            i =>
            {
                var leftVector = Avx.LoadVector256(leftBlockPointer + i);
                var rightVector = Avx.LoadVector256(leftBlockPointer + i);
                var mask = Avx.CompareGreaterThanOrEqual(leftVector, rightVector);
                var blended = Avx.BlendVariable(zeros, ones, mask);

                blended.Store(resultBlockPointer + i);
            });

        for (var i = done; i < n; i++)
        {
            resultBlockPointer[i] = leftBlockPointer[i] >= rightBlockPointer[i] ? 1d : 0d;
        }
    }
}