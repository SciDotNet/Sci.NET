// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedLinearAlgebraBackend : ILinearAlgebraBackend
{
    public void MatrixMultiply<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            result.Rows,
            0,
            result.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) =>
            {
                var sum = TNumber.Zero;

                for (var k = 0; k < left.Columns; k++)
                {
                    sum += leftMemoryBlock[(i * left.Columns) + k] *
                           rightMemoryBlock[(k * right.Columns) + j];
                }

                resultMemoryBlock[(i * result.Columns) + j] = sum;
            });
    }

    public void InnerProduct<TNumber>(Tensors.Vector<TNumber> left, Tensors.Vector<TNumber> right, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Handle;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Handle;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            left.Length,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            i => resultMemoryBlock[0] += leftMemoryBlock[i] * rightMemoryBlock[i]);
    }
}