// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.LinearAlgebra.Implementations;

internal class VectorOperationsService : IVectorOperationsService
{
    public Scalar<TNumber> CosineSimilarity<TNumber>(Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, IFloatingPoint<TNumber>, IRootFunctions<TNumber>
    {
        var dotProduct = left.Inner(right);
        var leftNorm = left.Norm();
        var rightNorm = right.Norm();

        return dotProduct.Divide(leftNorm.Multiply(rightNorm));
    }

    public Scalar<TNumber> CosineDistance<TNumber>(Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, IFloatingPoint<TNumber>, IRootFunctions<TNumber>
    {
        using var one = new Scalar<TNumber>(TNumber.One);
        using var minusOne = new Scalar<TNumber>(TNumber.Zero - TNumber.One);

        one.To(left.Device);
        minusOne.To(left.Device);

        return CosineSimilarity(left, right).Subtract(one).Multiply(minusOne);
    }

    public Scalar<TNumber> Norm<TNumber>(Vector<TNumber> vector)
        where TNumber : unmanaged, IRootFunctions<TNumber>, IFloatingPoint<TNumber>
    {
        return vector.Inner(vector).Sqrt();
    }
}