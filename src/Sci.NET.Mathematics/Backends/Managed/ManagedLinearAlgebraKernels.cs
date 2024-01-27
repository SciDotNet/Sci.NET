// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Numerics.Intrinsics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedLinearAlgebraKernels : ILinearAlgebraKernels
{
    public void MatrixMultiply<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Memory;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Memory;
        var kLength = left.Columns;
        var jLength = result.Columns;
        var vectorCount = SimdVector.Count<TNumber>();

        _ = LazyParallelExecutor.For(
            0,
            result.Rows,
            0,
            result.Columns,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            (i, j) =>
            {
                var sum = TNumber.Zero;
                var k = 0;

                for (; k <= kLength - vectorCount; k += vectorCount)
                {
                    var leftVector = leftMemoryBlock.UnsafeGetVectorUnchecked<TNumber>((i * kLength) + k);
                    var rightVector = rightMemoryBlock.UnsafeGetVectorUnchecked<TNumber>((k * jLength) + j);

                    sum += leftVector.Dot(rightVector);
                }

                for (; k < kLength; k++)
                {
                    sum += leftMemoryBlock[(i * kLength) + k] * rightMemoryBlock[(k * jLength) + j];
                }

                resultMemoryBlock[(i * jLength) + j] = sum;
            });
    }

    public void InnerProduct<TNumber>(Tensors.Vector<TNumber> left, Tensors.Vector<TNumber> right, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMemoryBlock = (SystemMemoryBlock<TNumber>)left.Memory;
        var rightMemoryBlock = (SystemMemoryBlock<TNumber>)right.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TNumber>)result.Memory;
        var vectorCount = SimdVector.Count<TNumber>();
        var sums = new ConcurrentDictionary<long, ISimdVector<TNumber>>();
        var done = 0L;

        if (left.Length >= vectorCount)
        {
            done = LazyParallelExecutor.For(
                0,
                left.Length - vectorCount,
                ManagedTensorBackend.ParallelizationThreshold / 2,
                vectorCount,
                i =>
                {
                    var leftVector = leftMemoryBlock.UnsafeGetVectorUnchecked<TNumber>(i);
                    var rightVector = rightMemoryBlock.UnsafeGetVectorUnchecked<TNumber>(i);

                    _ = sums.AddOrUpdate(
                        i / vectorCount,
                        _ => leftVector.Multiply(rightVector),
                        (_, sum) => sum.Add(leftVector.Multiply(rightVector)));
                });
        }

        for (var i = done; i < left.Length; i++)
        {
            resultMemoryBlock[0] += leftMemoryBlock[i] * rightMemoryBlock[i];
        }

        if (sums.Values.Count > 0)
        {
            resultMemoryBlock[0] += sums.Values.Aggregate((x, y) => x.Add(y)).Sum();
        }
    }
}