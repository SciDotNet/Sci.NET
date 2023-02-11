// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Backends;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.Elementwise;

/// <summary>
/// Extension methods for scalar multiplication of a <see cref="ITensor{TNumber}"/>.
/// </summary>
[PublicAPI]
public static class ScalarProductExtensions
{
    /// <summary>
    /// Performs a scalar multiplication of the left and right operands.
    /// </summary>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the operation.</returns>
    /// <exception cref="InvalidShapeException">Throws when the shapes are invalid for the scalar multiply operation.</exception>
    public static ITensor<TNumber> ScalarProduct<TNumber>(this ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
#pragma warning disable S2234, IDE0046

        if (!left.IsScalar && !right.IsScalar)
        {
            throw new InvalidShapeException(
                $"The shapes of the operands ({left.GetShape()} and {right.GetShape()}) are incompatible with the scalar product operation.");
        }

        return !left.IsScalar
            ? TensorBackend.Instance.ScalarMultiply(right, left)
            : TensorBackend.Instance.ScalarMultiply(left, right);

#pragma warning restore S2234, IDE0046
    }

    /// <summary>
    /// Performs a scalar multiplication of the left and right operands.
    /// </summary>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the operation.</returns>
    public static ITensor<TNumber> ScalarProduct<TNumber>(this ITensor<TNumber> left, TNumber right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorBackend.Instance.ScalarMultiply(right, left);
    }

    /// <summary>
    /// Performs a matrix multiplication of the left and right matrices.
    /// </summary>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the operation.</returns>
    public static ITensor<TNumber> MatrixMultiply<TNumber>(this ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorBackend.Instance.LinearAlgebra.MatrixMultiply(left, right);
    }
}