// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Backends;

/// <summary>
/// Linear algebra methods for <see cref="ITensor{TNumber}"/>.
/// </summary>
[PublicAPI]
public interface ILinearAlgebraBackendOperations
{
    /// <summary>
    /// Performs a matrix multiplication of the two tensors.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the matrix multiplication between the two operands.</returns>
    public ITensor<TNumber> MatrixMultiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// For vectors, the inner product of two <see cref="ITensor{TNumber}"/>s is calculated,
    /// for higher dimensions then the sum product over the last axes are calculated.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the inner product operation.</returns>
    /// <exception cref="ArgumentException">Throws when the operand shapes are incompatible with the
    /// inner product operation.</exception>
    public ITensor<TNumber> InnerProduct<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;
}