// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for a backend that provides power operations.
/// </summary>
[PublicAPI]
public interface IPowerKernels
{
    /// <summary>
    /// Raises <paramref name="value"/> to the power of <paramref name="power"/> and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <param name="result">The result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Pow<TNumber>(ITensor<TNumber> value, Scalar<TNumber> power, ITensor<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Raises <paramref name="value"/> to the power of -1 and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="value">The value to raise to the power of -1.</param>
    /// <param name="power">The power to raise the value to.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result in.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void PowDerivative<TNumber>(ITensor<TNumber> value, Scalar<TNumber> power, ITensor<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Raises a <see cref="Tensor{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result in.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Square<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Raises e to the power of <paramref name="value"/> and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="value">The exponent.</param>
    /// <param name="result">The <see cref="Tensor{TNumber}"/> to store the result in.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Exp<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Finds the natural logarithm of <paramref name="value"/> and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="value">The value to find the natural logarithm of.</param>
    /// <param name="result">The <see cref="Scalar{TNumber}"/> to store the result in.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Log<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Finds the derivative of the natural logarithm of <paramref name="value"/> and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="value">The value to find the derivative of the natural logarithm of.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result in.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void LogDerivative<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>;
}