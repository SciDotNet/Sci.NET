// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Equality;

/// <summary>
/// Provides a set of operations to perform element-wise <see cref="ITensor{TNumber}"/> comparisons, including equality,
/// greater than, less than, and their respective inclusive comparisons.
/// </summary>
[PublicAPI]
public interface ITensorEqualityOperationService
{
    /// <summary>
    /// Compares two <see cref="ITensor{TNumber}"/>s element-wise for equality and returns a <see cref="ITensor{TNumber}"/>
    /// containing the results of the comparison for each corresponding element.
    /// </summary>
    /// <param name="left">The left-hand operand <see cref="ITensor{TNumber}"/> for the comparison.</param>
    /// <param name="right">The right-hand operand <see cref="ITensor{TNumber}"/> for the comparison.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="ITensor{TNumber}"/> elements.</typeparam>
    /// <returns>
    /// A tensor containing the results of the element-wise comparison, where
    /// each element is resolved based on the equality of the corresponding elements
    /// from the input <see cref="ITensor{TNumber}"/>s, where equal elements are <see cref="INumberBase{TNumber}.One"/>
    /// and unequal elements are <see cref="INumberBase{TNumber}.Zero"/>.
    /// </returns>
    public ITensor<TNumber> PointwiseEquals<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Compares two <see cref="ITensor{TNumber}"/>s element-wise for inequality and returns a <see cref="ITensor{TNumber}"/>
    /// containing the results of the comparison for each corresponding element.
    /// </summary>
    /// <param name="left">The left-hand operand <see cref="ITensor{TNumber}"/> for the comparison.</param>
    /// <param name="right">The right-hand operand <see cref="ITensor{TNumber}"/> for the comparison.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="ITensor{TNumber}"/> elements.</typeparam>
    /// <returns>
    /// A tensor containing the results of the element-wise comparison, where
    /// each element is resolved based on the inequality of the corresponding elements
    /// from the input <see cref="ITensor{TNumber}"/>s, where unequal elements are <see cref="INumberBase{TNumber}.One"/>
    /// and equal elements are <see cref="INumberBase{TNumber}.Zero"/>.
    /// </returns>
    public ITensor<TNumber> PointwiseNotEqual<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Compares two <see cref="ITensor{TNumber}"/>s element-wise to determine whether the elements of the
    /// left-hand operand are strictly greater than the corresponding elements of the right-hand operand,
    /// and returns a <see cref="ITensor{TNumber}"/> containing the results of the comparison.
    /// </summary>
    /// <param name="left">The left-hand operand <see cref="ITensor{TNumber}"/> for the comparison.</param>
    /// <param name="right">The right-hand operand <see cref="ITensor{TNumber}"/> for the comparison.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="ITensor{TNumber}"/> elements.</typeparam>
    /// <returns>
    /// A tensor containing the results of the element-wise comparison, where each element
    /// is resolved based on whether the corresponding element from the left-hand operand
    /// is strictly greater than the corresponding element from the right-hand operand.
    /// If an element of the left-hand operand is greater, the result is <see cref="INumberBase{TNumber}.One"/>,
    /// otherwise <see cref="INumberBase{TNumber}.Zero"/>.
    /// </returns>
    public ITensor<TNumber> PointwiseGreaterThan<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Compares two <see cref="ITensor{TNumber}"/>s element-wise to check if each element of the left-hand operand is
    /// greater than or equal to the corresponding element of the right-hand operand, and returns a
    /// <see cref="ITensor{TNumber}"/> containing the results of the comparison.
    /// </summary>
    /// <param name="left">The left-hand operand <see cref="ITensor{TNumber}"/> for the comparison.</param>
    /// <param name="right">The right-hand operand <see cref="ITensor{TNumber}"/> for the comparison.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="ITensor{TNumber}"/> elements.</typeparam>
    /// <returns>
    /// A tensor containing the results of the element-wise comparison, where each element is resolved based on
    /// whether the corresponding element in the left operand is greater than or equal to the corresponding
    /// element in the right operand, represented as <see cref="INumberBase{TNumber}.One"/> for true and
    /// <see cref="INumberBase{TNumber}.Zero"/> for false.
    /// </returns>
    public ITensor<TNumber> PointwiseGreaterThanOrEqual<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Compares two <see cref="ITensor{TNumber}"/>s element-wise to determine if each element in the left-hand operand
    /// is less than the corresponding element in the right-hand operand, and returns a <see cref="ITensor{TNumber}"/>
    /// containing the results of the comparison.
    /// </summary>
    /// <param name="left">The left-hand operand <see cref="ITensor{TNumber}"/> for the comparison.</param>
    /// <param name="right">The right-hand operand <see cref="ITensor{TNumber}"/> for the comparison.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="ITensor{TNumber}"/> elements.</typeparam>
    /// <returns>
    /// A tensor containing the results of the element-wise comparison, where
    /// each element is resolved based on whether the corresponding element from the left-hand operand
    /// is less than the corresponding element from the right-hand operand. If true, the value is <see cref="INumberBase{TNumber}.One"/>;
    /// otherwise, the value is <see cref="INumberBase{TNumber}.Zero"/>.
    /// </returns>
    public ITensor<TNumber> PointwiseLessThan<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Compares two <see cref="ITensor{TNumber}"/>s element-wise to determine if each element in the left tensor
    /// is less than or equal to the corresponding element in the right tensor and returns a <see cref="ITensor{TNumber}"/>
    /// containing the results of the comparison for each corresponding element.
    /// </summary>
    /// <param name="left">The left-hand operand <see cref="ITensor{TNumber}"/> for the comparison.</param>
    /// <param name="right">The right-hand operand <see cref="ITensor{TNumber}"/> for the comparison.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="ITensor{TNumber}"/> elements.</typeparam>
    /// <returns>
    /// A tensor containing the results of the element-wise comparison, where
    /// each element is resolved based on whether the corresponding element in the left tensor
    /// is less than or equal to the corresponding element in the right tensor.
    /// Equal or lesser elements are represented as <see cref="INumberBase{TNumber}.One"/>,
    /// and elements that do not satisfy the condition are represented as <see cref="INumberBase{TNumber}.Zero"/>.
    /// </returns>
    public ITensor<TNumber> PointwiseLessThanOrEqual<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;
}