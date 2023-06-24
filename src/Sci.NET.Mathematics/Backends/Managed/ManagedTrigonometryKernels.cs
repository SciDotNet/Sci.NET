// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedTrigonometryKernels : ITrigonometryKernels
{
    public void Sin<TNumber>(Scalar<TNumber> scalar, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var scalarBlock = (SystemMemoryBlock<TNumber>)scalar.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        resultBlock[0] = TNumber.Sin(scalarBlock[0]);
    }

    public void Sin<TNumber>(Tensors.Vector<TNumber> vector, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var vectorBlock = (SystemMemoryBlock<TNumber>)vector.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            vectorBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            i => resultBlock[i] = TNumber.Sin(vectorBlock[i]));
    }

    public void Sin<TNumber>(Matrix<TNumber> matrix, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var matrixBlock = (SystemMemoryBlock<TNumber>)matrix.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            matrix.Rows,
            0,
            matrix.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) => resultBlock[(i * result.Columns) + j] =
                TNumber.Sin(matrixBlock[(i * matrix.Columns) + j]));
    }

    public void Sin<TNumber>(Tensor<TNumber> tensor, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            tensorBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Sin(tensorBlock[i]));
    }

    public void Cos<TNumber>(Scalar<TNumber> scalar, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var scalarBlock = (SystemMemoryBlock<TNumber>)scalar.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        resultBlock[0] = TNumber.Cos(scalarBlock[0]);
    }

    public void Cos<TNumber>(Tensors.Vector<TNumber> vector, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var vectorBlock = (SystemMemoryBlock<TNumber>)vector.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            vectorBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            i => resultBlock[i] = TNumber.Cos(vectorBlock[i]));
    }

    public void Cos<TNumber>(Matrix<TNumber> matrix, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var matrixBlock = (SystemMemoryBlock<TNumber>)matrix.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            matrix.Rows,
            0,
            matrix.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) => resultBlock[(i * result.Columns) + j] =
                TNumber.Cos(matrixBlock[(i * matrix.Columns) + j]));
    }

    public void Cos<TNumber>(Tensor<TNumber> tensor, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            tensorBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Cos(tensorBlock[i]));
    }

    public void Tan<TNumber>(Scalar<TNumber> scalar, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var scalarBlock = (SystemMemoryBlock<TNumber>)scalar.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        resultBlock[0] = TNumber.Tan(scalarBlock[0]);
    }

    public void Tan<TNumber>(Tensors.Vector<TNumber> vector, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var vectorBlock = (SystemMemoryBlock<TNumber>)vector.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            vectorBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            i => resultBlock[i] = TNumber.Tan(vectorBlock[i]));
    }

    public void Tan<TNumber>(Matrix<TNumber> matrix, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var matrixBlock = (SystemMemoryBlock<TNumber>)matrix.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            matrix.Rows,
            0,
            matrix.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) => resultBlock[(i * result.Columns) + j] =
                TNumber.Tan(matrixBlock[(i * matrix.Columns) + j]));
    }

    public void Tan<TNumber>(Tensor<TNumber> tensor, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var tensorBlock = (SystemMemoryBlock<TNumber>)tensor.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            tensorBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Tan(tensorBlock[i]));
    }

    public void Sinh<TNumber>(Scalar<TNumber> scalar, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var scalarBlock = (SystemMemoryBlock<TNumber>)scalar.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        resultBlock[0] = TNumber.Tanh(scalarBlock[0]);
    }

    public void Sinh<TNumber>(Tensors.Vector<TNumber> vector, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var vectorBlock = (SystemMemoryBlock<TNumber>)vector.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            vectorBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Tanh(vectorBlock[i]));
    }

    public void Sinh<TNumber>(Matrix<TNumber> matrix, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var matrixBlock = (SystemMemoryBlock<TNumber>)matrix.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            matrix.Rows,
            0,
            matrix.Columns,
            ManagedTensorBackend.ParallelizationThreshold,
            (i, j) => resultBlock[(i * result.Columns) + j] =
                TNumber.Tanh(matrixBlock[(i * matrix.Columns) + j]));
    }

    public void Sinh<TNumber>(Tensor<TNumber> tensor, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var vectorBlock = (SystemMemoryBlock<TNumber>)tensor.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            vectorBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Tanh(vectorBlock[i]));
    }
}