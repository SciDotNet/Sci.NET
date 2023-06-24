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
    public void Pow<TNumber>(Scalar<TNumber> value, Scalar<TNumber> power, Scalar<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Raises <paramref name="value"/> to the power of <paramref name="power"/> and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <param name="result">The result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Pow<TNumber>(Tensors.Vector<TNumber> value, Scalar<TNumber> power, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Raises <paramref name="value"/> to the power of <paramref name="power"/> and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <param name="result">The result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Pow<TNumber>(Matrix<TNumber> value, Scalar<TNumber> power, Matrix<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Raises <paramref name="value"/> to the power of <paramref name="power"/> and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <param name="result">The result.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    public void Pow<TNumber>(Tensor<TNumber> value, Scalar<TNumber> power, Tensor<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>;
}