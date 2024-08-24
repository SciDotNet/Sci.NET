// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.LinearAlgebra.Implementations;

internal class MatrixMultiplicationService : IMatrixMultiplicationService
{
    private readonly IDeviceGuardService _guardService;

    public MatrixMultiplicationService(ITensorOperationServiceProvider provider)
    {
        _guardService = provider.GetDeviceGuardService();
    }

    public Matrix<TNumber> MatrixMultiply<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, bool? overrideRequiresGradient = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);

        if (left.Shape.Dimensions[1] != right.Shape.Dimensions[0])
        {
            throw new InvalidShapeException(
                $"The number of columns of the left matrix must match the number of rows of the right matrix but got {left.Shape} and {right.Shape}.");
        }

        var result = new Matrix<TNumber>(left.Shape.Dimensions[0], right.Shape.Dimensions[1], left.Backend, overrideRequiresGradient ?? (left.RequiresGradient || right.RequiresGradient));

        left.Backend.LinearAlgebra.MatrixMultiply(left, right, result);

        if (overrideRequiresGradient ?? left.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                left,
                grad => grad.ToMatrix().MatrixMultiply(right.Transpose()));
        }

        if (overrideRequiresGradient ?? right.RequiresGradient)
        {
            ((ITensor<TNumber>)result).AddParent(
                right,
                grad => left.Transpose().MatrixMultiply(grad.ToMatrix()));
        }

        return result;
    }
}