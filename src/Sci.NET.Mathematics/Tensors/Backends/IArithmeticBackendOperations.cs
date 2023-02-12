// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Backends;

/// <summary>
/// The interface for arithmetic operations.
/// </summary>
[PublicAPI]
public interface IArithmeticBackendOperations
{
    /// <summary>
    /// Adds two tensors.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The sum of the two operands.</returns>
    public ITensor<TNumber> Add<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Adds a number to a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The sum of the two operands.</returns>
    public ITensor<TNumber> Add<TNumber>(TNumber left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Subtracts the right operand from the left operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The difference between the right operand and the left operand.</returns>
    public ITensor<TNumber> Subtract<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Subtracts the right operand from the left operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The difference between the right operand and the left operand.</returns>
    public ITensor<TNumber> Subtract<TNumber>(TNumber left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Subtracts the right operand from the left operand.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The difference between the right operand and the left operand.</returns>
    public ITensor<TNumber> Subtract<TNumber>(ITensor<TNumber> left, TNumber right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Negates the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to negate.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The negated <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Negate<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The quotient of the left and right operands.</returns>
    public ITensor<TNumber> Divide<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The quotient of the left and right operands.</returns>
    public ITensor<TNumber> Divide<TNumber>(TNumber left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The quotient of the left and right operands.</returns>
    public ITensor<TNumber> Divide<TNumber>(ITensor<TNumber> left, TNumber right)
        where TNumber : unmanaged, INumber<TNumber>;
}