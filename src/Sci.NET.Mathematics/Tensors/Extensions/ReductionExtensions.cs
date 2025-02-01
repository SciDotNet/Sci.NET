// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;

#pragma warning disable IDE0130

// ReSharper disable once CheckNamespace
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// Provides extension methods for <see cref="ITensor{TNumber}"/> reduction operations.
/// </summary>
[PublicAPI]
public static class ReductionExtensions
{
    /// <summary>
    /// Reduces a <see cref="ITensor{TNumber}"/> to a given shape.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to reduce.</param>
    /// <param name="targetShape">The shape to reduce the <see cref="ITensor{TNumber}"/> to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The reduced <see cref="ITensor{TNumber}"/>.</returns>
    public static ITensor<TNumber> ReduceToShape<TNumber>(this ITensor<TNumber> tensor, Shape targetShape)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetReductionService()
            .ReduceToShape(tensor, targetShape);
    }

    /// <summary>
    /// Determines if a <see cref="ITensor{TNumber}"/> can be reduced to a given shape.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to reduce.</param>
    /// <param name="shape">The shape to reduce the <see cref="ITensor{TNumber}"/> to.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>A value indicating whether the <see cref="ITensor{TNumber}"/> can be reduced to the given shape.</returns>
    public static bool CanReduceToShape<TNumber>(this ITensor<TNumber> tensor, Shape shape)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetReductionService()
            .CanReduceToShape(tensor, shape);
    }

    /// <summary>
    /// Computes the sum of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to sum.</param>
    /// <param name="axes">The axes to sum over.</param>
    /// <param name="keepDims">A value indicating whether the dimensions should be preserved.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The sum of all of the elements in the <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Sum<TNumber>(this ITensor<TNumber> tensor, int[]? axes = null, bool keepDims = false)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetReductionService()
            .Sum(tensor, axes, keepDims);
    }

    /// <summary>
    /// Computes the mean of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to find the mean of.</param>
    /// <param name="axes">The axes to find the mean over.</param>
    /// <param name="keepDims">A value indicating whether the dimensions should be preserved.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The mean of the <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Mean<TNumber>(this ITensor<TNumber> tensor, int[]? axes = null, bool keepDims = false)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetReductionService()
            .Mean(tensor, axes, keepDims);
    }

    /// <summary>
    /// Computes the maximum value of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to find the maximum value of.</param>
    /// <param name="axes">The axes to find the maximum value over.</param>
    /// <param name="keepDims">A value indicating whether the dimensions should be preserved.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The maximum value of the <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Max<TNumber>(this ITensor<TNumber> tensor, int[]? axes = null, bool keepDims = false)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetReductionService()
            .Max(tensor, axes, keepDims);
    }

    /// <summary>
    /// Computes the minimum value of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to find the minimum value of.</param>
    /// <param name="axes">The axes to find the minimum value over.</param>
    /// <param name="keepDims">A value indicating whether the dimensions should be preserved.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The minimum value of the <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Min<TNumber>(this ITensor<TNumber> tensor, int[]? axes = null, bool keepDims = false)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetReductionService()
            .Min(tensor, axes, keepDims);
    }
}