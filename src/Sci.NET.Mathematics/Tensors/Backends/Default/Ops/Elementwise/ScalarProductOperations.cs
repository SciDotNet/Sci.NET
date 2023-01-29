// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Backends.Default.Ops.Elementwise;

internal static class ScalarProductOperations
{
    public static unsafe Tensor<TNumber> ScalarProduct<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (left.Rank != 0 && right.Rank != 0)
        {
            throw new ArgumentException("At least one of the tensors must be a scalar.");
        }

        var scalar = left.Rank == 0 ? left : right;
        var tensor = left.Rank == 0 ? right : left;

        var result = new Tensor<TNumber>(new Shape(tensor.Dimensions));
        var scalarPtr = scalar.Data[0];
        var tensorPtr = tensor.Data;
        var resultPtr = result.Data;

        for (var i = 0; i < tensor.ElementCount; i++)
        {
            resultPtr[i] = scalarPtr * tensorPtr[i];
        }

        return result;
    }
}