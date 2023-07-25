// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;
using Sci.NET.Mathematics.Tensors.Exceptions;

// ReSharper disable once CheckNamespace
#pragma warning disable IDE0130 // API accessibility
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// Matrix multiplication extensions for <see cref="ITensor{TNumber}"/>.
/// </summary>
[PublicAPI]
public static class MatrixMultiplicationExtensions
{
    /// <summary>
    /// Performs a matrix multiplication between two rank-2 tensors (matrices).
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The result of the matrix multiplication.</returns>
    /// <exception cref="InvalidShapeException">The shapes of the operands were invalid.</exception>
    [DebuggerStepThrough]
    public static Matrix<TNumber> MatrixMultiply<TNumber>(this Matrix<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetMatrixMultiplicationService()
            .MatrixMultiply(left, right);
    }
}