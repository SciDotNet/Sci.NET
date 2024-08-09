// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Numerics;
using Sci.NET.Common.Numerics.Intrinsics;

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
        var vectorCount = SimdVector.Count<TNumber>();

        var i = LazyParallelExecutor.For(
            0,
            n - vectorCount,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorCount,
            i =>
            {
                var leftVector = leftBlock.UnsafeGetVectorUnchecked<TNumber>(i);
                var rightVector = rightBlock.UnsafeGetVectorUnchecked<TNumber>(i);
                var resultVector = leftVector.Add(rightVector);

                resultBlock.UnsafeSetVectorUnchecked(resultVector, i);
            });

        for (; i < n; i++)
        {
            resultBlock[i] = leftBlock[i] + rightBlock[i];
        }
    }

    public void AddTensorTensorInplace<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)left;
        var rightBlock = (SystemMemoryBlock<TNumber>)right;
        var vectorCount = SimdVector.Count<TNumber>();

        var i = LazyParallelExecutor.For(
            0,
            n - vectorCount,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorCount,
            i =>
            {
                var leftVector = leftBlock.UnsafeGetVectorUnchecked<TNumber>(i);
                var rightVector = rightBlock.UnsafeGetVectorUnchecked<TNumber>(i);
                var resultVector = leftVector.Add(rightVector);

                leftBlock.UnsafeSetVectorUnchecked(resultVector, i);
            });

        for (; i < n; i++)
        {
            leftBlock[i] += rightBlock[i];
        }
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

        _ = LazyParallelExecutor.For(
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

        _ = LazyParallelExecutor.For(
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
        var vectorCount = SimdVector.Count<TNumber>();

        var i = LazyParallelExecutor.For(
            0,
            n - vectorCount,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorCount,
            i =>
            {
                var leftVector = leftBlock.UnsafeGetVectorUnchecked<TNumber>(i);
                var rightVector = rightBlock.UnsafeGetVectorUnchecked<TNumber>(i);
                var resultVector = leftVector.Subtract(rightVector);

                resultBlock.UnsafeSetVectorUnchecked(resultVector, i);
            });

        for (; i < n; i++)
        {
            resultBlock[i] = leftBlock[i] - rightBlock[i];
        }
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

        _ = LazyParallelExecutor.For(
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

        _ = LazyParallelExecutor.For(
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
        var vectorCount = SimdVector.Count<TNumber>();

        var i = LazyParallelExecutor.For(
            0,
            n - vectorCount,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorCount,
            i =>
            {
                var leftVector = leftBlock.UnsafeGetVectorUnchecked<TNumber>(i);
                var rightVector = rightBlock.UnsafeGetVectorUnchecked<TNumber>(i);
                var resultVector = leftVector.Multiply(rightVector);

                resultBlock.UnsafeSetVectorUnchecked(resultVector, i);
            });

        for (; i < n; i++)
        {
            resultBlock[i] = leftBlock[i] * rightBlock[i];
        }
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

        _ = LazyParallelExecutor.For(
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

        _ = LazyParallelExecutor.For(
            0,
            m,
            0,
            n,
            ManagedTensorBackend.ParallelizationThreshold,
            (i, j) => resultBlock[(i * n) + j] = leftBlock[j] * rightBlock[(i * n) + j]);
    }

    public void MultiplyTensorTensorInplace<TNumber>(IMemoryBlock<TNumber> leftMemory, IMemoryBlock<TNumber> rightMemory, long shapeElementCount)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var left = (SystemMemoryBlock<TNumber>)leftMemory;
        var right = (SystemMemoryBlock<TNumber>)rightMemory;
        var vectorCount = SimdVector.Count<TNumber>();

        var i = LazyParallelExecutor.For(
            0,
            shapeElementCount - vectorCount,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorCount,
            i =>
            {
                var leftVector = left.UnsafeGetVectorUnchecked<TNumber>(i);
                var rightVector = right.UnsafeGetVectorUnchecked<TNumber>(i);
                var resultVector = leftVector.Multiply(rightVector);

                left.UnsafeSetVectorUnchecked(resultVector, i);
            });

        for (; i < shapeElementCount; i++)
        {
            left[i] *= right[i];
        }
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
        var vectorCount = SimdVector.Count<TNumber>();

        var i = LazyParallelExecutor.For(
            0,
            n - vectorCount,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorCount,
            i =>
            {
                var leftVector = leftBlock.UnsafeGetVectorUnchecked<TNumber>(i);
                var rightVector = rightBlock.UnsafeGetVectorUnchecked<TNumber>(i);
                var resultVector = leftVector.Divide(rightVector);

                resultBlock.UnsafeSetVectorUnchecked(resultVector, i);
            });

        for (; i < n; i++)
        {
            resultBlock[i] = leftBlock[i] / rightBlock[i];
        }
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

        _ = LazyParallelExecutor.For(
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

        _ = LazyParallelExecutor.For(
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
        var vectorCount = SimdVector.Count<TNumber>();

        var i = LazyParallelExecutor.For(
            0,
            n - vectorCount,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorCount,
            i =>
            {
                var tensorVector = tensorBlock.UnsafeGetVectorUnchecked<TNumber>(i);

                tensorVector = tensorVector.Negate();

                resultBlock.UnsafeSetVectorUnchecked(tensorVector, i);
            });

        for (; i < n; i++)
        {
            resultBlock[i] = -tensorBlock[i];
        }
    }

    public void Abs<TNumber>(
        IMemoryBlock<TNumber> tensor,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;
        var vectorCount = SimdVector.Count<TNumber>();

        var i = LazyParallelExecutor.For(
            0,
            n - vectorCount,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorCount,
            i =>
            {
                var tensorVector = tensorBlock.UnsafeGetVectorUnchecked<TNumber>(i);
                tensorVector = tensorVector.Abs();
                resultBlock.UnsafeSetVectorUnchecked(tensorVector, i);
            });

        for (; i < n; i++)
        {
            resultBlock[i] = TNumber.Abs(tensorBlock[i]);
        }
    }

    public void AbsoluteDifference<TNumber>(IMemoryBlock<TNumber> left, IMemoryBlock<TNumber> right, IMemoryBlock<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftBlock = (SystemMemoryBlock<TNumber>)left;
        var rightBlock = (SystemMemoryBlock<TNumber>)right;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;
        var vectorCount = SimdVector.Count<TNumber>();

        var i = LazyParallelExecutor.For(
            0,
            resultBlock.Length - vectorCount,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorCount,
            i =>
            {
                var leftVector = leftBlock.UnsafeGetVectorUnchecked<TNumber>(i);
                var rightVector = rightBlock.UnsafeGetVectorUnchecked<TNumber>(i);
                var resultVector = leftVector.Subtract(rightVector).Abs();

                resultBlock.UnsafeSetVectorUnchecked(resultVector, i);
            });

        for (; i < resultBlock.Length; i++)
        {
            resultBlock[i] = TNumber.Abs(leftBlock[i] - rightBlock[i]);
        }
    }

    public void Sqrt<TNumber>(
        IMemoryBlock<TNumber> tensor,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor;
        var resultBlock = (SystemMemoryBlock<TNumber>)result;
        var vectorCount = SimdVector.Count<TNumber>();

        var i = LazyParallelExecutor.For(
            0,
            n - vectorCount,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorCount,
            i =>
            {
                var tensorVector = tensorBlock.UnsafeGetVectorUnchecked<TNumber>(i);

                tensorVector = tensorVector.Sqrt();

                resultBlock.UnsafeSetVectorUnchecked(tensorVector, i);
            });

        i *= vectorCount;

        for (; i < n; i++)
        {
            resultBlock[i] = GenericMath.Sqrt(tensorBlock[i]);
        }
    }
}