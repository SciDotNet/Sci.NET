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
    public ITensor<TNumber> Pow<TNumber>(ITensor<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Raises a <see cref="Scalar{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The <paramref name="value"/> raised to the second power.</returns>
    public ITensor<TNumber> Square<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Raises e to the power of <paramref name="value"/>.
    /// </summary>
    /// <param name="value">The value to raise e to the power of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>e raised to the power of <paramref name="value"/>.</returns>
    public ITensor<TNumber> Exp<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Finds the natural logarithm of <paramref name="value"/>.
    /// </summary>
    /// <param name="value">The value to find the natural logarithm of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The natural logarithm of <paramref name="value"/>.</returns>
    public ITensor<TNumber> Log<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>;
}