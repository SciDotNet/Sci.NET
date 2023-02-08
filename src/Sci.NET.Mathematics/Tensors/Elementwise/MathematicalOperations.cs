// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Backends;

namespace Sci.NET.Mathematics.Tensors.Elementwise;

/// <summary>
/// Extension methods for mathematical operations on <see cref="ITensor{TNumber}"/>.
/// </summary>
[PublicAPI]
public static class MathematicalOperations
{
    /// <summary>
    /// Calculates the element-wise exponential of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="input">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The element-wise exponential of the input <see cref="ITensor{TNumber}"/>.</returns>
    public static ITensor<TNumber> Exp<TNumber>(this ITensor<TNumber> input)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        return TensorBackend.Instance.MathematicalOperations.Exp(input);
    }
}