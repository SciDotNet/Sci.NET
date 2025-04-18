// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.LinearAlgebra.Implementations;

internal class MatrixMultiplicationService : IMatrixMultiplicationService
{
    private readonly IDeviceGuardService _guardService;
    private readonly IGradientAppenderService _gradientAppenderService;

    public MatrixMultiplicationService()
    {
        _guardService = TensorServiceProvider.GetTensorOperationServiceProvider().GetDeviceGuardService();
        _gradientAppenderService = TensorServiceProvider.GetTensorOperationServiceProvider().GetGradientAppenderService();
    }

    public Matrix<TNumber> MatrixMultiply<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, bool? overrideRequiresGradient = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var backend = _guardService.GuardBinaryOperation(left.Device, right.Device);

        if (left.Shape.Dimensions[1] != right.Shape.Dimensions[0])
        {
            throw new InvalidShapeException(
                $"The number of columns of the left matrix must match the number of rows of the right matrix but got {left.Shape} and {right.Shape}.");
        }

        var result = new Matrix<TNumber>(left.Shape.Dimensions[0], right.Shape.Dimensions[1], backend, overrideRequiresGradient ?? (left.RequiresGradient || right.RequiresGradient));

        backend.LinearAlgebra.MatrixMultiply(left, right, result);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            left,
            right,
            overrideRequiresGradient,
            grad =>
            {
                var matrixMultiplicationService = TensorServiceProvider.GetTensorOperationServiceProvider().GetMatrixMultiplicationService();
                using var gradMatrix = grad.ToMatrix();
                using var rightTransposed = right.Transpose();
                var resultGrad = matrixMultiplicationService.MatrixMultiply(gradMatrix, rightTransposed, false);

                return ((ITensor<TNumber>)resultGrad).AsGradient();
            },
            grad =>
            {
                var matrixMultiplicationService = TensorServiceProvider.GetTensorOperationServiceProvider().GetMatrixMultiplicationService();
                using var gradMatrix = grad.ToMatrix();
                using var leftTransposed = left.Transpose();
                var resultGrad = matrixMultiplicationService.MatrixMultiply(leftTransposed, gradMatrix, false);

                return ((ITensor<TNumber>)resultGrad).AsGradient();
            });

        return result;
    }
}