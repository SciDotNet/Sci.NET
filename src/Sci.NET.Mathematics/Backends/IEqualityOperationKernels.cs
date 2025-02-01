// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for a backend that provides equality operations.
/// </summary>
[PublicAPI]
public interface IEqualityOperationKernels
{
    /// <summary>
    /// Compares two <see cref="IMemoryBlock{TNumber}"/>s element-wise for equality and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="leftOperand">The left operand.</param>
    /// <param name="rightOperand">The right operand.</param>
    /// <param name="result">The result.</param>
    /// <param name="n">The number of elements to compare.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>.</typeparam>
    public void PointwiseEqualsKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Compares two <see cref="IMemoryBlock{TNumber}"/>s element-wise for inequality and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="leftOperand">The left operand.</param>
    /// <param name="rightOperand">The right operand.</param>
    /// <param name="result">The result.</param>
    /// <param name="n">The number of elements to compare.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>.</typeparam>
    public void PointwiseNotEqualKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Compares two <see cref="IMemoryBlock{TNumber}"/>s element-wise to determine if the elements of
    /// <paramref name="leftOperand"/> are greater than those of <paramref name="rightOperand"/>, and stores
    /// the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="leftOperand">The left operand.</param>
    /// <param name="rightOperand">The right operand.</param>
    /// <param name="result">The result.</param>
    /// <param name="n">The number of elements to compare.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>.</typeparam>
    public void PointwiseGreaterThanKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Compares two <see cref="IMemoryBlock{TNumber}"/>s element-wise to determine if the elements in the left operand
    /// are greater than or equal to the corresponding elements in the right operand, and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="leftOperand">The left operand.</param>
    /// <param name="rightOperand">The right operand.</param>
    /// <param name="result">The result.</param>
    /// <param name="n">The number of elements to compare.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>.</typeparam>
    public void PointwiseGreaterThanOrEqualKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Compares two <see cref="IMemoryBlock{TNumber}"/>s element-wise to check if the left operand is less than the right operand and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="leftOperand">The left operand.</param>
    /// <param name="rightOperand">The right operand.</param>
    /// <param name="result">The result.</param>
    /// <param name="n">The number of elements to compare.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>.</typeparam>
    public void PointwiseLessThanKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Compares two <see cref="IMemoryBlock{TNumber}"/>s element-wise to check if the elements of the left operand are less than or equal to those of the right operand, and stores the result in <paramref name="result"/>.
    /// </summary>
    /// <param name="leftOperand">The left operand.</param>
    /// <param name="rightOperand">The right operand.</param>
    /// <param name="result">The result.</param>
    /// <param name="n">The number of elements to compare.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="IMemoryBlock{TNumber}"/>.</typeparam>
    public void PointwiseLessThanOrEqualKernel<TNumber>(IMemoryBlock<TNumber> leftOperand, IMemoryBlock<TNumber> rightOperand, IMemoryBlock<TNumber> result, long n)
        where TNumber : unmanaged, INumber<TNumber>;
}