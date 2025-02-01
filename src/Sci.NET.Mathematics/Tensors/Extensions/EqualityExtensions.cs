// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;

// ReSharper disable once CheckNamespace
#pragma warning disable IDE0130
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// A set of extension methods for equality operations.
/// </summary>
[PublicAPI]
public static class EqualityExtensions
{
    /// <summary>
    /// Performs a pointwise equality operation on two <see cref="ITensor{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The result of the pointwise equality operation.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> PointwiseEquals<TNumber>(this ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseEquals(left, right);
    }

    /// <summary>
    /// Performs a pointwise equality operation between two <see cref="Scalar{TNumber}"/> instances.
    /// </summary>
    /// <param name="left">The left operand as a <see cref="Scalar{TNumber}"/>.</param>
    /// <param name="right">The right operand as a <see cref="Scalar{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>A <see cref="Scalar{TNumber}"/> representing the result of the pointwise equality operation.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> PointwiseEquals<TNumber>(this Scalar<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseEquals(left, right)
            .ToScalar();
    }

    /// <summary>
    /// Performs a pointwise equality operation on two <see cref="Vector{TNumber}"/>s, comparing each element individually.
    /// </summary>
    /// <param name="left">The left <see cref="Vector{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Vector{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The numeric type of the elements in the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>A <see cref="Vector{TNumber}"/> representing the result of the pointwise equality operation.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> PointwiseEquals<TNumber>(this Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseEquals(left, right)
            .ToVector();
    }

    /// <summary>
    /// Performs a pointwise equality operation on two <see cref="Matrix{TNumber}"/> instances.
    /// </summary>
    /// <param name="left">The left <see cref="Matrix{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Matrix{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The numeric type of the elements in the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>A new <see cref="Matrix{TNumber}"/> representing the result of the pointwise equality operation.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> PointwiseEquals<TNumber>(this Matrix<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseEquals(left, right)
            .ToMatrix();
    }

    /// <summary>
    /// Performs a pointwise equality operation on two <see cref="Tensor{TNumber}"/> instances.
    /// </summary>
    /// <param name="left">The left <see cref="Tensor{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Tensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>A new <see cref="Tensor{TNumber}"/> representing the result of the pointwise equality operation.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> PointwiseEquals<TNumber>(this Tensor<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseEquals(left, right)
            .ToTensor();
    }

    /// <summary>
    /// Performs a pointwise inequality operation on two <see cref="ITensor{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The result of the pointwise inequality operation.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> PointwiseNotEquals<TNumber>(this ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseNotEqual(left, right);
    }

    /// <summary>
    /// Performs a pointwise inequality operation on two <see cref="ITensor{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The result of the pointwise inequality operation.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> PointwiseNotEquals<TNumber>(this Scalar<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseNotEqual(left, right)
            .ToScalar();
    }

    /// <summary>
    /// Performs a pointwise inequality operation on two <see cref="Vector{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="Vector{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Vector{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>s.</typeparam>
    /// <returns>The result of the pointwise inequality operation as a <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> PointwiseNotEquals<TNumber>(this Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseNotEqual(left, right)
            .ToVector();
    }

    /// <summary>
    /// Performs a pointwise inequality operation on two <see cref="Matrix{TNumber}"/> instances.
    /// </summary>
    /// <param name="left">The left <see cref="Matrix{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Matrix{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/> instances.</typeparam>
    /// <returns>The result of the pointwise inequality operation as a <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> PointwiseNotEquals<TNumber>(this Matrix<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseNotEqual(left, right)
            .ToMatrix();
    }

    /// <summary>
    /// Performs a pointwise inequality operation on two <see cref="ITensor{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The result of the pointwise inequality operation.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> PointwiseNotEquals<TNumber>(this Tensor<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseNotEqual(left, right)
            .ToTensor();
    }

    /// <summary>
    /// Performs a pointwise comparison to determine if the elements in the left <see cref="ITensor{TNumber}"/>
    /// are greater than the corresponding elements in the right <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left <see cref="ITensor{TNumber}"/> to be compared.</param>
    /// <param name="right">The right <see cref="ITensor{TNumber}"/> to compare against.</param>
    /// <typeparam name="TNumber">The numeric type of the elements within the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>A new <see cref="ITensor{TNumber}"/> where each element indicates whether the corresponding
    /// element in the left tensor is greater than the element in the right tensor.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> PointwiseGreaterThan<TNumber>(this ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseGreaterThan(left, right);
    }

    /// <summary>
    /// Performs a pointwise "greater than" comparison on two <see cref="Scalar{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="Scalar{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Scalar{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="Scalar{TNumber}"/>s.</typeparam>
    /// <returns>A <see cref="Scalar{TNumber}"/> representing the result of the pointwise "greater than" comparison.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> PointwiseGreaterThan<TNumber>(this Scalar<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseGreaterThan(left, right)
            .ToScalar();
    }

    /// <summary>
    /// Performs a pointwise greater-than operation on two <see cref="Vector{TNumber}"/> instances.
    /// </summary>
    /// <param name="left">The left <see cref="Vector{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Vector{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="Vector{TNumber}"/>s.</typeparam>
    /// <returns>The result of the pointwise greater-than operation as a <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> PointwiseGreaterThan<TNumber>(this Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseGreaterThan(left, right)
            .ToVector();
    }

    /// <summary>
    /// Performs a pointwise greater-than operation on two <see cref="Matrix{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="Matrix{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Matrix{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>s.</typeparam>
    /// <returns>The result of the pointwise greater-than operation as a <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> PointwiseGreaterThan<TNumber>(this Matrix<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseGreaterThan(left, right)
            .ToMatrix();
    }

    /// <summary>
    /// Performs a pointwise greater-than comparison between two <see cref="Tensor{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="Tensor{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Tensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>s.</typeparam>
    /// <returns>A <see cref="Tensor{TNumber}"/> representing the result of the pointwise greater-than comparison.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> PointwiseGreaterThan<TNumber>(this Tensor<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseGreaterThan(left, right)
            .ToTensor();
    }

    /// <summary>
    /// Performs a pointwise greater than or equal operation on two <see cref="ITensor{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The result of the pointwise greater than or equal operation.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> PointwiseGreaterThanOrEqual<TNumber>(this ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseGreaterThanOrEqual(left, right);
    }

    /// <summary>
    /// Performs a pointwise greater-than-or-equal-to comparison operation on two <see cref="Scalar{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="Scalar{TNumber}"/> operand.</param>
    /// <param name="right">The right <see cref="Scalar{TNumber}"/> operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>s.</typeparam>
    /// <returns>The result of the pointwise greater-than-or-equal-to operation as a <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> PointwiseGreaterThanOrEqual<TNumber>(this Scalar<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseGreaterThanOrEqual(left, right)
            .ToScalar();
    }

    /// <summary>
    /// Performs a pointwise "greater than or equal to" operation on two <see cref="Vector{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="Vector{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Vector{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>s.</typeparam>
    /// <returns>The result of the pointwise "greater than or equal to" operation as a <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> PointwiseGreaterThanOrEqual<TNumber>(this Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseGreaterThanOrEqual(left, right)
            .ToVector();
    }

    /// <summary>
    /// Performs a pointwise greater-than-or-equal-to operation on two <see cref="Matrix{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="Matrix{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Matrix{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="Matrix{TNumber}"/>s.</typeparam>
    /// <returns>A <see cref="Matrix{TNumber}"/> representing the result of the pointwise greater-than-or-equal-to operation.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> PointwiseGreaterThanOrEqual<TNumber>(this Matrix<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseGreaterThanOrEqual(left, right)
            .ToMatrix();
    }

    /// <summary>
    /// Performs a pointwise greater than or equal to operation on two <see cref="Tensor{TNumber}"/> instances.
    /// </summary>
    /// <param name="left">The left <see cref="Tensor{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Tensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="Tensor{TNumber}"/> instances.</typeparam>
    /// <returns>A new <see cref="Tensor{TNumber}"/> representing the result of the operation.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> PointwiseGreaterThanOrEqual<TNumber>(this Tensor<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseGreaterThanOrEqual(left, right)
            .ToTensor();
    }

    /// <summary>
    /// Performs a pointwise less-than comparison between two <see cref="ITensor{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>A new <see cref="ITensor{TNumber}"/> representing the result of the pointwise less-than comparison.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> PointwiseLessThan<TNumber>(this ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseLessThan(left, right);
    }

    /// <summary>
    /// Performs a pointwise "less than" comparison between two <see cref="Scalar{TNumber}"/> values.
    /// </summary>
    /// <param name="left">The left <see cref="Scalar{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Scalar{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>s.</typeparam>
    /// <returns>A <see cref="Scalar{TNumber}"/> representing the result of the comparison.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> PointwiseLessThan<TNumber>(this Scalar<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseLessThan(left, right)
            .ToScalar();
    }

    /// <summary>
    /// Performs a pointwise less-than comparison on two <see cref="Vector{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="Vector{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Vector{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>s.</typeparam>
    /// <returns>A <see cref="Vector{TNumber}"/> representing the result of the pointwise less-than comparison.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> PointwiseLessThan<TNumber>(this Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseLessThan(left, right)
            .ToVector();
    }

    /// <summary>
    /// Performs a pointwise less-than comparison between two <see cref="Matrix{TNumber}"/> instances.
    /// </summary>
    /// <param name="left">The left <see cref="Matrix{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Matrix{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The numeric type of the elements in the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>A <see cref="Matrix{TNumber}"/> containing the results of the pointwise less-than comparison.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> PointwiseLessThan<TNumber>(this Matrix<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseLessThan(left, right)
            .ToMatrix();
    }

    /// <summary>
    /// Performs a pointwise less-than operation on two <see cref="Tensor{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="Tensor{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Tensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>s.</typeparam>
    /// <returns>The result of the pointwise less-than operation as a <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> PointwiseLessThan<TNumber>(this Tensor<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseLessThan(left, right)
            .ToTensor();
    }

    /// <summary>
    /// Performs a pointwise less-than-or-equal operation on two <see cref="ITensor{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The result of the pointwise less-than-or-equal operation.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> PointwiseLessThanOrEqual<TNumber>(this ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseLessThanOrEqual(left, right);
    }

    /// <summary>
    /// Performs a pointwise "less than or equal to" operation on two <see cref="Scalar{TNumber}"/> instances.
    /// </summary>
    /// <param name="left">The left <see cref="Scalar{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Scalar{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The numeric type of the <see cref="Scalar{TNumber}"/> instances.</typeparam>
    /// <returns>The result of the pointwise "less than or equal to" operation as a <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> PointwiseLessThanOrEqual<TNumber>(this Scalar<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseLessThanOrEqual(left, right)
            .ToScalar();
    }

    /// <summary>
    /// Performs a pointwise "less than or equal to" operation on two <see cref="Vector{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="Vector{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Vector{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>s.</typeparam>
    /// <returns>The result of the pointwise "less than or equal to" operation.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> PointwiseLessThanOrEqual<TNumber>(this Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseLessThanOrEqual(left, right)
            .ToVector();
    }

    /// <summary>
    /// Performs a pointwise less-than-or-equal-to comparison between two <see cref="Matrix{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="Matrix{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Matrix{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The numeric type of the elements in the <see cref="Matrix{TNumber}"/>s.</typeparam>
    /// <returns>A <see cref="Matrix{TNumber}"/> representing the result of the comparison.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> PointwiseLessThanOrEqual<TNumber>(this Matrix<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseLessThanOrEqual(left, right)
            .ToMatrix();
    }

    /// <summary>
    /// Performs a pointwise "less than or equal to" operation on two <see cref="Tensor{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="Tensor{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Tensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>s.</typeparam>
    /// <returns>The result of the pointwise "less than or equal to" operation.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> PointwiseLessThanOrEqual<TNumber>(this Tensor<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetEqualityOperationService()
            .PointwiseLessThanOrEqual(left, right)
            .ToTensor();
    }
}