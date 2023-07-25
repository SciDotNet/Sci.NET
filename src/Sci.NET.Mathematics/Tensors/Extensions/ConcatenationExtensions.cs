// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;

#pragma warning disable IDE0130

// ReSharper disable once CheckNamespace
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// Tensor concatenation extension methods for <see cref="ITensor{TNumber}"/>.
/// </summary>
[PublicAPI]
public static class ConcatenationExtensions
{
    /// <summary>
    /// Concatenates the collection of <see cref="Scalar{TNumber}"/> into a <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="scalars">The collection of <see cref="Scalar{TNumber}"/> to concatenate.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The result of the concatenation operation.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Concatenate<TNumber>(this ICollection<Scalar<TNumber>> scalars)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetConcatenationService()
            .Concatenate(scalars);
    }

    /// <summary>
    /// Concatenates the collection of <see cref="Vector{TNumber}"/> into a <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="vectors">The collection of <see cref="Vector{TNumber}"/> to concatenate.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The result of the concatenation operation.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Concatenate<TNumber>(this ICollection<Vector<TNumber>> vectors)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetConcatenationService()
            .Concatenate(vectors);
    }

    /// <summary>
    /// Concatenates the collection of <see cref="Matrix{TNumber}"/> into a <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="matrices">The collection of <see cref="Matrix{TNumber}"/> to concatenate.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The result of the concatenation operation.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Concatenate<TNumber>(this ICollection<Matrix<TNumber>> matrices)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetConcatenationService()
            .Concatenate(matrices);
    }

    /// <summary>
    /// Concatenates the collection of <see cref="Tensor{TNumber}"/> into a <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensors">The collection of <see cref="Tensor{TNumber}"/> to concatenate.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the concatenation operation.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Concatenate<TNumber>(this ICollection<Tensor<TNumber>> tensors)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetConcatenationService()
            .Concatenate(tensors);
    }
}