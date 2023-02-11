// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Performance;

namespace Sci.NET.Mathematics.Tensors.Backends.Default;

internal class DefaultLinearAlgebraBackendOperations : ILinearAlgebraBackendOperations
{
    public ITensor<TNumber> MatrixMultiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Rank != 2 || right.Rank != 2)
        {
            throw new ArgumentException("Both tensors must be matrices.");
        }

        if (left.Dimensions[1] != right.Dimensions[0])
        {
            throw new ArgumentException(
                "The last dimension of the left tensor must be equal to the first dimension of the right tensor.");
        }

        var result = new Tensor<TNumber>(new Shape(left.Dimensions[0], right.Dimensions[1]));

        LazyParallelExecutor.For(
            0,
            left.Dimensions[0],
            DefaultTensorBackend.ParallelizationThreshold / 2,
            i =>
            {
                LazyParallelExecutor.For(
                    0,
                    right.Dimensions[1],
                    DefaultTensorBackend.ParallelizationThreshold / 2,
                    j =>
                    {
                        var sum = TNumber.Zero;

                        for (var k = 0; k < left.Dimensions[1]; k++)
                        {
                            sum += left.Data[(i * left.Dimensions[1]) + k] * right.Data[(k * right.Dimensions[1]) + j];
                        }

                        result.Data[(i * result.Dimensions[1]) + j] = sum;
                    });
            });

        return result;
    }

    public ITensor<TNumber> InnerProduct<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Rank != 1 || right.Rank != 1)
        {
            throw new ArgumentException("Both tensors must be vectors.");
        }

        if (left.Dimensions[0] != right.Dimensions[0])
        {
            throw new ArgumentException("The last dimension of both tensors must be equal.");
        }

        var result = new Tensor<TNumber>(new Shape(0));

        LazyParallelExecutor.For(
            0,
            left.ElementCount,
            DefaultTensorBackend.ParallelizationThreshold,
            i => result.Data[0] += left.Data[i] * right.Data[i]);

        return result;
    }
}