// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.LinearAlgebra.Implementations;

internal class MatrixMultiplicationService : IMatrixMultiplicationService
{
    private readonly IDeviceGuardService _guardService;

    public MatrixMultiplicationService(ITensorOperationServiceFactory factory)
    {
        _guardService = factory.GetDeviceGuardService();
    }

    public Matrix<TNumber> MatrixMultiply<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        _guardService.GuardBinaryOperation(left.Device, right.Device);

        if (left.Shape[0] != right.Shape[1])
        {
            throw new InvalidShapeException(
                "The number of columns of the left matrix must match the number of rows of the right matrix.");
        }

        var resultShape = new Shape(left.Rows, right.Columns);
        var resultMemory = left.Backend.Storage.Allocate<TNumber>(resultShape);
        var result = new Matrix<TNumber>(resultShape[0], resultShape[1], resultMemory, left.Backend);

        left.Backend.LinearAlgebra.MatrixMultiply(left, right, result);

        return result;
    }
}