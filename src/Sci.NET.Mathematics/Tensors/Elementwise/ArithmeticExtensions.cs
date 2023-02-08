// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Backends;

namespace Sci.NET.Mathematics.Tensors.Elementwise;

/// <summary>
/// Extension methods for elementwise arithmetic operations.
/// </summary>
[PublicAPI]
public static class ArithmeticExtensions
{
    /// <summary>
    /// Adds two tensors.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The sum of the two operands.</returns>
    public static ITensor<TNumber> Add<TNumber>(this ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorBackend.Instance.Arithmetic.Add(left, right);
    }

    /// <summary>
    /// Subtracts the right operand from the left operand..
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The difference between the right operand and the left operand.</returns>
    public static ITensor<TNumber> Subtract<TNumber>(this ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorBackend.Instance.Arithmetic.Subtract(left, right);
    }
}