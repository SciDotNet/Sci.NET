// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for arithmetic kernels.
/// </summary>
[PublicAPI]
public interface IArithmeticKernels
{
    /// <summary>
    /// Finds the element-wise sum of two <see cref="IMemoryBlock{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="IMemoryBlock{TNumber}"/> to add.</param>
    /// <param name="right">The right <see cref="IMemoryBlock{TNumber}"/> to add.</param>
    /// <param name="result">The result of the addition.</param>
    /// <param name="n">The number of elements to add.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>s.</typeparam>
    public void AddTensorTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Adds the elements of the right <see cref="IMemoryBlock{TNumber}"/> to the left <see cref="IMemoryBlock{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left <see cref="IMemoryBlock{TNumber}"/> to add to.</param>
    /// <param name="right">The right <see cref="IMemoryBlock{TNumber}"/> to add.</param>
    /// <param name="n">The number of elements to add.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>s.</typeparam>
    public void AddTensorTensorInplace<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Adds the elements of Adds the elements of two <see cref="IMemoryBlock{TNumber}"/>s together.
    /// </summary>
    /// <param name="left">The left <see cref="IMemoryBlock{TNumber}"/> to add.</param>
    /// <param name="right">The right <see cref="IMemoryBlock{TNumber}"/> to add.</param>
    /// <param name="result">The result of the addition.</param>
    /// <param name="m">The number of elements to add in the common dimensions.</param>
    /// <param name="n">The number of elements to add in the broadcast dimensions.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>s.</typeparam>
    public void AddTensorBroadcastTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Adds the elements of two <see cref="IMemoryBlock{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="IMemoryBlock{TNumber}"/> to add.</param>
    /// <param name="right">The right <see cref="IMemoryBlock{TNumber}"/> to add.</param>
    /// <param name="result">The result of the addition.</param>
    /// <param name="m">The number of elements to add in the common dimensions.</param>
    /// <param name="n">The number of elements to add in the broadcast dimensions.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>s.</typeparam>
    public void AddBroadcastTensorTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Subtracts the elements of two <see cref="IMemoryBlock{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="IMemoryBlock{TNumber}"/> to add.</param>
    /// <param name="right">The right <see cref="IMemoryBlock{TNumber}"/> to add.</param>
    /// <param name="result">The result of the addition.</param>
    /// <param name="n">The number of elements to add.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>s.</typeparam>
    public void SubtractTensorTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Subtracts the elements of two <see cref="IMemoryBlock{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="IMemoryBlock{TNumber}"/> to subtract from.</param>
    /// <param name="right">The right <see cref="IMemoryBlock{TNumber}"/> to subtract.</param>
    /// <param name="result">The result of the subtraction.</param>
    /// <param name="m">The number of elements to subtract in the common dimensions.</param>
    /// <param name="n">The number of elements to subtract in the broadcast dimensions.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>s.</typeparam>
    public void SubtractTensorBroadcastTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Subtracts the elements of two <see cref="IMemoryBlock{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="IMemoryBlock{TNumber}"/> to subtract from.</param>
    /// <param name="right">The right <see cref="IMemoryBlock{TNumber}"/> to subtract.</param>
    /// <param name="result">The result of the subtraction.</param>
    /// <param name="m">The number of elements to subtract in the common dimensions.</param>
    /// <param name="n">The number of elements to subtract in the broadcast dimensions.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>s.</typeparam>
    public void SubtractBroadcastTensorTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Multiplies the elements of two <see cref="IMemoryBlock{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="IMemoryBlock{TNumber}"/> to multiply.</param>
    /// <param name="right">The right <see cref="IMemoryBlock{TNumber}"/> to multiply.</param>
    /// <param name="result">The result of the multiplication.</param>
    /// <param name="n">The number of elements to multiply.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>s.</typeparam>
    public void MultiplyTensorTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Multiplies the elements of two <see cref="IMemoryBlock{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="IMemoryBlock{TNumber}"/> to multiply.</param>
    /// <param name="right">The right <see cref="IMemoryBlock{TNumber}"/> to multiply.</param>
    /// <param name="result">The result of the multiplication.</param>
    /// <param name="m">The number of elements to multiply in the common dimensions.</param>
    /// <param name="n">The number of elements to multiply in the broadcast dimensions.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>s.</typeparam>
    public void MultiplyTensorBroadcastTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Multiplies the elements of two <see cref="IMemoryBlock{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="IMemoryBlock{TNumber}"/> to multiply.</param>
    /// <param name="right">The right <see cref="IMemoryBlock{TNumber}"/> to multiply.</param>
    /// <param name="result">The result of the multiplication.</param>
    /// <param name="m">The number of elements to multiply in the common dimensions.</param>
    /// <param name="n">The number of elements to multiply in the broadcast dimensions.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>s.</typeparam>
    public void MultiplyBroadcastTensorTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Multiplies the elements of two <see cref="IMemoryBlock{TNumber}"/>s in place.
    /// </summary>
    /// <param name="leftMemory">The left <see cref="IMemoryBlock{TNumber}"/> to multiply.</param>
    /// <param name="rightMemory">The right <see cref="IMemoryBlock{TNumber}"/> to multiply.</param>
    /// <param name="shapeElementCount">The number of elements to multiply.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>s.</typeparam>
    public void MultiplyTensorTensorInplace<TNumber>(IMemoryBlock<TNumber> leftMemory, IMemoryBlock<TNumber> rightMemory, long shapeElementCount)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Divides the elements of two <see cref="IMemoryBlock{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="IMemoryBlock{TNumber}"/> to divide.</param>
    /// <param name="right">The right <see cref="IMemoryBlock{TNumber}"/> to divide.</param>
    /// <param name="result">The result of the division.</param>
    /// <param name="n">The number of elements to divide.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>s.</typeparam>
    public void DivideTensorTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Divides the elements of two <see cref="IMemoryBlock{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="IMemoryBlock{TNumber}"/> to divide.</param>
    /// <param name="right">The right <see cref="IMemoryBlock{TNumber}"/> to divide.</param>
    /// <param name="result">The result of the division.</param>
    /// <param name="m">The number of elements to divide in the common dimensions.</param>
    /// <param name="n">The number of elements to divide in the broadcast dimensions.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>s.</typeparam>
    public void DivideTensorBroadcastTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Divides the elements of two <see cref="IMemoryBlock{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="IMemoryBlock{TNumber}"/> to divide.</param>
    /// <param name="right">The right <see cref="IMemoryBlock{TNumber}"/> to divide.</param>
    /// <param name="result">The result of the division.</param>
    /// <param name="m">The number of elements to divide in the common dimensions.</param>
    /// <param name="n">The number of elements to divide in the broadcast dimensions.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>s.</typeparam>
    public void DivideBroadcastTensorTensor<TNumber>(
        IMemoryBlock<TNumber> left,
        IMemoryBlock<TNumber> right,
        IMemoryBlock<TNumber> result,
        long m,
        long n)
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