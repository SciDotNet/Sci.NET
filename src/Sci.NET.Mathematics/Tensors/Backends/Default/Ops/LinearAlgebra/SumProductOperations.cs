// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Backends.Default.Ops.LinearAlgebra;

internal static class SumProductOperations
{
    public static unsafe ITensor<TNumber> InnerProduct<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
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

        var leftPtr = left.Handle.ToPointer();
        var rightPtr = right.Handle.ToPointer();
        var resultPtr = result.Handle.ToPointer();

        for (var i = 0; i < left.ElementCount; i++)
        {
            resultPtr[0] = leftPtr[i] * rightPtr[i];
        }

        return result;
    }
}