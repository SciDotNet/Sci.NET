// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedHyperbolicKernels : IHyperbolicKernels
{
    public void Sinh<TNumber>(Scalar<TNumber> scalar, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var scalarHandle = (SystemMemoryBlock<TNumber>)scalar.Handle;
        var resultHandle = (SystemMemoryBlock<TNumber>)result.Handle;

        resultHandle[0] = TNumber.Sinh(scalarHandle[0]);
    }

    public void Sinh<TNumber>(Tensors.Vector<TNumber> vector, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var vectorHandle = (SystemMemoryBlock<TNumber>)vector.Handle;
        var resultHandle = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            vector.Length,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            i => resultHandle[i] = TNumber.Sinh(vectorHandle[i]));
    }

    public void Sinh<TNumber>(Matrix<TNumber> matrix, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var matrixHandle = (SystemMemoryBlock<TNumber>)matrix.Handle;
        var resultHandle = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            matrix.Rows,
            0,
            matrix.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) => resultHandle[(i * result.Columns) + j] =
                TNumber.Sinh(matrixHandle[(i * matrix.Columns) + j]));
    }

    public void Sinh<TNumber>(Tensor<TNumber> tensor, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        var tensorHandle = (SystemMemoryBlock<TNumber>)tensor.Handle;
        var resultHandle = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            tensorHandle.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultHandle[i] = TNumber.Sinh(tensorHandle[i]));
    }
}