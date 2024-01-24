// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedLinearAlgebraKernels : ILinearAlgebraKernels
{
    [MethodImpl(MethodImplOptions.NoOptimization)]
    public void MatrixMultiply<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        // BUG: Vector<TNumber> has strange behaviour when sizeof(TNumber) > 8
        if (System.Numerics.Vector<TNumber>.IsSupported && Vector.IsHardwareAccelerated && Unsafe.SizeOf<TNumber>() < 8)
        {
            MatrixMultiplySimd(
                left,
                right,
                result);

            return;
        }

        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Memory;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Memory;

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
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Memory;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        LazyParallelExecutor.For(
            0,
            left.Length,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            i => resultMemoryBlock[0] += leftMemoryBlock[i] * rightMemoryBlock[i]);
    }

    private static void MatrixMultiplySimd<TNumber>(
        Matrix<TNumber> left,
        Matrix<TNumber> right,
        Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Memory;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Memory;
        var vectorSize = System.Numerics.Vector<TNumber>.Count;
        var kLength = left.Columns;
        var jLength = result.Columns;

        LazyParallelExecutor.For(
            0,
            result.Rows,
            0,
            result.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) =>
            {
                var sum = TNumber.Zero;
                var k = 0;

                for (; k <= kLength - vectorSize; k += vectorSize)
                {
                    var leftSpan = leftMemoryBlock.AsSpan((i * kLength) + k, vectorSize);
                    var rightSpan = rightMemoryBlock.AsSpan((k * jLength) + j, vectorSize);
                    var vLeft = new System.Numerics.Vector<TNumber>(leftSpan);
                    var vRight = new System.Numerics.Vector<TNumber>(rightSpan);

                    sum += Vector.Dot(vLeft, vRight);
                }

                for (; k < kLength; k++)
                {
                    sum += leftMemoryBlock[(i * kLength) + k] * rightMemoryBlock[(k * jLength) + j];
                }

                resultMemoryBlock[(i * jLength) + j] = sum;
            });
    }
}