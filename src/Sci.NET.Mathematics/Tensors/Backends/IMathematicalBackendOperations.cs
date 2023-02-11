// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Backends;

/// <summary>
/// Interface for mathematical operations for an <see cref="ITensor{TNumber}"/>.
/// </summary>
[PublicAPI]
public interface IMathematicalBackendOperations
{
    /// <summary>
    /// Calculates the element-wise exponential of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="input">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the input <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The exponential of the input <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Exp<TNumber>(ITensor<TNumber> input)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>;
}