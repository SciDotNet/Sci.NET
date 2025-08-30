// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for arithmetic kernels.
/// </summary>
[PublicAPI]
public interface IArithmeticKernels
{
    /// <summary>
    /// Performs element-wise addition of two <see cref="ITensor{TNumber}"/>s and stores the result in a third <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="right">The right operand <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result of the addition.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="ITensor{TNumber}"/> elements.</typeparam>
    public void Add<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs element-wise subtraction of two <see cref="ITensor{TNumber}"/>s and stores the result in a third <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="right">The right operand <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result of the subtraction.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="ITensor{TNumber}"/> elements.</typeparam>
    public void Subtract<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs element-wise multiplication of two <see cref="ITensor{TNumber}"/>s and stores the result in a third <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The <paramref name="left"/> operand <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="right">The <paramref name="right"/> operand <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result of the multiplication.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="ITensor{TNumber}"/> elements.</typeparam>
    public void Multiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs element-wise division of two <see cref="ITensor{TNumber}"/>s and stores the result in a third <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The <paramref name="left"/> operand <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="right">The <paramref name="right"/> operand <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="result">The <see cref="ITensor{TNumber}"/> to store the result of the division.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="ITensor{TNumber}"/> elements.</typeparam>
    public void Divide<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Negates the elements of a <see cref="IMemoryBlock{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="IMemoryBlock{TNumber}"/> to negate.</param>
    /// <param name="result">The result of the negation.</param>
    /// <param name="n">The number of elements to negate.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>.</typeparam>
    public void Negate<TNumber>(
        IMemoryBlock<TNumber> tensor,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Takes the absolute value of the elements of a <see cref="IMemoryBlock{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="IMemoryBlock{TNumber}"/> to take the absolute value of.</param>
    /// <param name="result">The result of the absolute value.</param>
    /// <param name="n">The number of elements to take the absolute value of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>.</typeparam>
    public void Abs<TNumber>(
        IMemoryBlock<TNumber> tensor,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Calculates the gradient of the absolute value of a <see cref="IMemoryBlock{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="IMemoryBlock{TNumber}"/> to take the absolute value of.</param>
    /// <param name="gradient">The incoming gradient.</param>
    /// <param name="result">The result of the gradient calculation.</param>
    /// <param name="n">The number of elements to calculate the gradient of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>.</typeparam>
    public void AbsGradient<TNumber>(
        IMemoryBlock<TNumber> tensor,
        IMemoryBlock<TNumber> gradient,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the absolute difference between two <see cref="IMemoryBlock{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="IMemoryBlock{TNumber}"/> to subtract from.</param>
    /// <param name="right">The right <see cref="IMemoryBlock{TNumber}"/> to subtract.</param>
    /// <param name="result">The result of the absolute difference.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>s.</typeparam>
    public void AbsoluteDifference<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Takes the square root of the elements of a <see cref="IMemoryBlock{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="IMemoryBlock{TNumber}"/> to take the square root of.</param>
    /// <param name="result">The result of the square root.</param>
    /// <param name="n">The number of elements to take the square root of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>.</typeparam>
    public void Sqrt<TNumber>(
        IMemoryBlock<TNumber> tensor,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>;
}