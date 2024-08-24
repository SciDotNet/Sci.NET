// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.LinearAlgebra;

/// <summary>
/// An interface for matrix multiplication operations.
/// </summary>
[PublicAPI]
public interface IMatrixMultiplicationService
{
    /// <summary>
    /// Performs a matrix multiplication between two rank-2 tensors (matrices).
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="overrideRequiresGradient">When not <see langword="null"/>, overrides whether the resulting tensor requires gradient.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The result of the matrix multiplication.</returns>
    /// <exception cref="InvalidShapeException">The shapes of the operands were invalid.</exception>
    public Matrix<TNumber> MatrixMultiply<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, bool? overrideRequiresGradient = null)
        where TNumber : unmanaged, INumber<TNumber>;
}