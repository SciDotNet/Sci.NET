// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Pointwise;

/// <summary>
/// Provides power operations for <see cref="ITensor{TNumber}"/>s.
/// </summary>
[PublicAPI]
public interface IPowerService
{
    /// <summary>
    /// Raises a <see cref="Scalar{TNumber}"/> to the power of a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The base (<paramref name="value"/>) raised to the given <paramref name="power"/>.</returns>
    public Scalar<TNumber> Pow<TNumber>(Scalar<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Raises a <see cref="Vector{TNumber}"/> to the power of a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The base (<paramref name="value"/>) raised to the given <paramref name="power"/>.</returns>
    public Vector<TNumber> Pow<TNumber>(Vector<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Raises a <see cref="Matrix{TNumber}"/> to the power of a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The base (<paramref name="value"/>) raised to the given <paramref name="power"/>.</returns>
    public Matrix<TNumber> Pow<TNumber>(Matrix<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Raises a <see cref="Tensor{TNumber}"/> to the power of a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The base (<paramref name="value"/>) raised to the given <paramref name="power"/>.</returns>
    public Tensor<TNumber> Pow<TNumber>(Tensor<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Raises a <see cref="Scalar{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The <paramref name="value"/> raised to the second power.</returns>
    public Scalar<TNumber> Square<TNumber>(Scalar<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Raises a <see cref="Vector{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The <paramref name="value"/> raised to the second power.</returns>
    public Vector<TNumber> Square<TNumber>(Vector<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Raises a <see cref="Matrix{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The <paramref name="value"/> raised to the second power.</returns>
    public Matrix<TNumber> Square<TNumber>(Matrix<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Raises a <see cref="Tensor{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The <paramref name="value"/> raised to the second power.</returns>
    public Tensor<TNumber> Square<TNumber>(Tensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;
}