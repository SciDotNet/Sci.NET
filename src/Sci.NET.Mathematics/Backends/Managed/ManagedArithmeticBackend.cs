// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedArithmeticBackend : IArithmeticBackend
{
    public void Add<TNumber>(Scalar<TNumber> left, Scalar<TNumber> right, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        resultMemoryBlock[0] = leftMemoryBlock[0] + rightMemoryBlock[0];
    }

    public void Add<TNumber>(Scalar<TNumber> left, Tensors.Vector<TNumber> right, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            right.Length,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            i => resultMemoryBlock[i] = leftMemoryBlock[0] + rightMemoryBlock[i]);
    }

    public void Add<TNumber>(Scalar<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            right.Rows,
            0,
            right.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) => resultMemoryBlock[(i * result.Columns) + j] =
                leftMemoryBlock[0] + rightMemoryBlock[(i * right.Columns) + j]);
    }

    public void Add<TNumber>(Scalar<TNumber> left, Tensor<TNumber> right, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            right.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            i => resultMemoryBlock[i] = leftMemoryBlock[0] + rightMemoryBlock[i]);
    }

    public void Add<TNumber>(Tensors.Vector<TNumber> left, Scalar<TNumber> right, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            left.Length,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            i => resultMemoryBlock[i] = leftMemoryBlock[i] + rightMemoryBlock[0]);
    }

    public void Add<TNumber>(
        Tensors.Vector<TNumber> left,
        Tensors.Vector<TNumber> right,
        Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            left.Length,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            i => resultMemoryBlock[i] = leftMemoryBlock[i] + rightMemoryBlock[i]);
    }

    public void Add<TNumber>(Tensors.Vector<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            right.Rows,
            0,
            right.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) => resultMemoryBlock[(i * result.Columns) + j] =
                leftMemoryBlock[i] + rightMemoryBlock[(i * right.Columns) + j]);
    }

    public void Add<TNumber>(Matrix<TNumber> left, Scalar<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            left.Rows,
            0,
            left.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) => resultMemoryBlock[(i * result.Columns) + j] =
                leftMemoryBlock[(i * left.Columns) + j] + rightMemoryBlock[0]);
    }

    public void Add<TNumber>(Matrix<TNumber> left, Tensors.Vector<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            left.Rows,
            0,
            left.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) => resultMemoryBlock[(i * result.Columns) + j] =
                leftMemoryBlock[(i * left.Columns) + j] + rightMemoryBlock[i]);
    }

    public void Add<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            left.Rows,
            0,
            left.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) => resultMemoryBlock[(i * result.Columns) + j] =
                leftMemoryBlock[(i * left.Columns) + j] + rightMemoryBlock[(i * right.Columns) + j]);
    }

    public void Add<TNumber>(Tensor<TNumber> left, Scalar<TNumber> right, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            left.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            i => resultMemoryBlock[i] = leftMemoryBlock[i] + rightMemoryBlock[0]);
    }

    public void Add<TNumber>(Tensor<TNumber> left, Tensor<TNumber> right, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            left.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            i => resultMemoryBlock[i] = leftMemoryBlock[i] + rightMemoryBlock[i]);
    }

    public void Subtract<TNumber>(Scalar<TNumber> left, Scalar<TNumber> right, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        resultMemoryBlock[0] = leftMemoryBlock[0] - rightMemoryBlock[0];
    }

    public void Subtract<TNumber>(Scalar<TNumber> left, Tensors.Vector<TNumber> right, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            right.Length,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            i => resultMemoryBlock[i] = leftMemoryBlock[0] - rightMemoryBlock[i]);
    }

    public void Subtract<TNumber>(Scalar<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            right.Rows,
            0,
            right.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) => resultMemoryBlock[(i * result.Columns) + j] =
                leftMemoryBlock[0] - rightMemoryBlock[(i * right.Columns) + j]);
    }

    public void Subtract<TNumber>(Scalar<TNumber> left, Tensor<TNumber> right, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            right.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultMemoryBlock[i] = leftMemoryBlock[0] - rightMemoryBlock[i]);
    }

    public void Subtract<TNumber>(Tensors.Vector<TNumber> left, Scalar<TNumber> right, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            left.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultMemoryBlock[i] = leftMemoryBlock[i] - rightMemoryBlock[0]);
    }

    public void Subtract<TNumber>(
        Tensors.Vector<TNumber> left,
        Tensors.Vector<TNumber> right,
        Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            left.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultMemoryBlock[i] = leftMemoryBlock[i] - rightMemoryBlock[i]);
    }

    public void Subtract<TNumber>(Tensors.Vector<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            right.Rows,
            0,
            right.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) => resultMemoryBlock[(i * result.Columns) + j] =
                leftMemoryBlock[i] - rightMemoryBlock[(i * right.Columns) + j]);
    }

    public void Subtract<TNumber>(Matrix<TNumber> left, Scalar<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            left.Rows,
            0,
            left.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) => resultMemoryBlock[(i * result.Columns) + j] =
                leftMemoryBlock[(i * left.Columns) + j] - rightMemoryBlock[0]);
    }

    public void Subtract<TNumber>(Matrix<TNumber> left, Tensors.Vector<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            left.Rows,
            0,
            left.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) => resultMemoryBlock[(i * result.Columns) + j] =
                leftMemoryBlock[(i * left.Columns) + j] - rightMemoryBlock[i]);
    }

    public void Subtract<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            left.Rows,
            0,
            left.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) => resultMemoryBlock[(i * result.Columns) + j] =
                leftMemoryBlock[(i * left.Columns) + j] - rightMemoryBlock[(i * right.Columns) + j]);
    }

    public void Subtract<TNumber>(Tensor<TNumber> left, Scalar<TNumber> right, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            left.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultMemoryBlock[i] = leftMemoryBlock[i] - rightMemoryBlock[0]);
    }

    public void Subtract<TNumber>(Tensor<TNumber> left, Tensor<TNumber> right, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            left.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultMemoryBlock[i] = leftMemoryBlock[i] - rightMemoryBlock[i]);
    }

    public unsafe void Multiply<TNumber>(Scalar<TNumber> left, Scalar<TNumber> right, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = ((SystemMemoryBlock<TNumber>)left.Handle).ToPointer()[0];
        var rightMemoryBlock = ((SystemMemoryBlock<TNumber>)right.Handle).ToPointer()[0];
        var resultMemoryBlock = ((SystemMemoryBlock<TNumber>)result.Handle).ToPointer();

        resultMemoryBlock[0] = leftMemoryBlock * rightMemoryBlock;
    }

    public unsafe void Multiply<TNumber>(
        Scalar<TNumber> left,
        Tensors.Vector<TNumber> right,
        Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = ((SystemMemoryBlock<TNumber>)left.Handle).ToPointer()[0];
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            right.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultMemoryBlock[i] = leftMemoryBlock * rightMemoryBlock[i]);
    }

    public unsafe void Multiply<TNumber>(Scalar<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = ((SystemMemoryBlock<TNumber>)left.Handle).ToPointer()[0];
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            right.Rows,
            0,
            right.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) => resultMemoryBlock[(i * result.Columns) + j] =
                leftMemoryBlock * rightMemoryBlock[(i * right.Columns) + j]);
    }

    public unsafe void Multiply<TNumber>(Scalar<TNumber> left, Tensor<TNumber> right, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = ((SystemMemoryBlock<TNumber>)left.Handle).ToPointer()[0];
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            right.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultMemoryBlock[i] = leftMemoryBlock * rightMemoryBlock[i]);
    }

    public void Divide<TNumber>(Scalar<TNumber> left, Scalar<TNumber> right, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        resultMemoryBlock[0] = leftMemoryBlock[0] / rightMemoryBlock[0];
    }

    public unsafe void Divide<TNumber>(Scalar<TNumber> left, Tensors.Vector<TNumber> right, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = ((SystemMemoryBlock<TNumber>)left.Handle).ToPointer()[0];
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            right.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultMemoryBlock[i] = leftMemoryBlock / rightMemoryBlock[i]);
    }

    public unsafe void Divide<TNumber>(Scalar<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = ((SystemMemoryBlock<TNumber>)left.Handle).ToPointer()[0];
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            right.Rows,
            0,
            right.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) => resultMemoryBlock[(i * result.Columns) + j] =
                leftMemoryBlock / rightMemoryBlock[(i * right.Columns) + j]);
    }

    public unsafe void Divide<TNumber>(Scalar<TNumber> left, Tensor<TNumber> right, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = ((SystemMemoryBlock<TNumber>)left.Handle).ToPointer()[0];
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            right.Shape.ElementCount,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultMemoryBlock[i] = leftMemoryBlock / rightMemoryBlock[i]);
    }
}