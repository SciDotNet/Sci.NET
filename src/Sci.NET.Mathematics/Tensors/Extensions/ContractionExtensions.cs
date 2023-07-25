// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;
using Sci.NET.Mathematics.Tensors.Exceptions;

// ReSharper disable once CheckNamespace
#pragma warning disable IDE0130 // API accessibility
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// Tensor contraction extension methods for <see cref="ITensor{TNumber}"/>.
/// </summary>
[PublicAPI]
public static class ContractionExtensions
{
    /// <summary>
    /// Performs a tensor contraction on the specified tensors.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="leftIndices">The indices of the left operand to contract over.</param>
    /// <param name="rightIndices">The indices of the right operand to contract over.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the contraction operation.</returns>
    /// <exception cref="ArgumentException">Throws when an argument is invalid.</exception>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Contract<TNumber>(
        this ITensor<TNumber> left,
        ITensor<TNumber> right,
        int[] leftIndices,
        int[] rightIndices)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetContractionService()
            .Contract(left, right, leftIndices, rightIndices);
    }

    /// <inheritdoc cref="Contract{TNumber}"/>
    /// <remarks>An alias of <see cref="Contract{TNumber}"/>.</remarks>
    [DebuggerStepThrough]
    public static ITensor<TNumber> TensorDot<TNumber>(
        this ITensor<TNumber> left,
        ITensor<TNumber> right,
        int[] leftIndices,
        int[] rightIndices)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetContractionService()
            .Contract(left, right, leftIndices, rightIndices);
    }

    /// <summary>
    /// For vectors, the inner product of two <see cref="ITensor{TNumber}"/>s is calculated,
    /// for higher dimensions then the sum product over the last axes are calculated.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the inner product operation.</returns>
    /// <exception cref="ArgumentException">Throws when the operand shapes are incompatible with the
    /// inner product operation.</exception>
    /// <exception cref="InvalidShapeException">The given shapes were not compatible with the inner product operation.</exception>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Inner<TNumber>(this Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetContractionService()
            .Inner(left, right);
    }

    /// <summary>
    /// Calculates the dot product of two <see cref="ITensor{TNumber}"/>s.
    /// <list type="bullet">
    /// <item>If both <paramref name="left"/> and <paramref name="right"/> are 1-D arrays, it is equivalent to <see cref="Inner{TNumber}"/>.</item>
    /// <item>If both <paramref name="left"/> and <paramref name="right"/> are 2-D arrays, it is equivalent to <see cref="MatrixMultiplicationExtensions.MatrixMultiply{TNumber}"/>.</item>
    /// <item>If either <paramref name="left"/> or <paramref name="right"/> is 0-D (scalar), it is equivalent to <see cref="ArithmeticExtensions.Multiply{TNumber}(Sci.NET.Mathematics.Tensors.Scalar{TNumber},Sci.NET.Mathematics.Tensors.Scalar{TNumber})"/></item>
    /// <item>If <paramref name="left"/> is an N-D array and <paramref name="right"/> is a 1-D array, it is a <see cref="Contract{TNumber}"/> operation over the last axis of <paramref name="left"/> and <paramref name="right"/>.</item>
    /// <item>If <paramref name="left"/> is an N-D array and <paramref name="right"/> is an M-D array (where M>=2), it is a <see cref="Contract{TNumber}"/> operation over the last axis of <paramref name="left"/> and the second-to-last axis of <paramref name="right"/>.</item>
    /// </list>
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the dot product operation.</returns>
    /// <exception cref="ArgumentException">Throws when the operand shapes are incompatible with the dot product operation.</exception>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Dot<TNumber>(
        this ITensor<TNumber> left,
        ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetContractionService()
            .Dot(left, right);
    }
}