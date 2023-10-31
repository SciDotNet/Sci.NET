// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;

#pragma warning disable IDE0130

// ReSharper disable once CheckNamespace
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// Provides extension methods for <see cref="Vector{TNumber}"/> operations.
/// </summary>
public static class VectorOperationExtensions
{
    /// <summary>
    /// Computes the cosine distance between two <see cref="Vector{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left <see cref="Vector{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="Vector{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>s.</typeparam>
    /// <returns>The cosine distance between the two <see cref="Vector{TNumber}"/>s.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> CosineDistance<TNumber>(this Vector<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, IFloatingPoint<TNumber>, IRootFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetVectorOperationsService()
            .CosineDistance(left, right);
    }

    /// <summary>
    /// Computes the norm of a <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="vector">The <see cref="Vector{TNumber}"/> to compute the norm of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The norm of the <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Norm<TNumber>(this Vector<TNumber> vector)
        where TNumber : unmanaged, IFloatingPoint<TNumber>, IRootFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetVectorOperationsService()
            .Norm(vector);
    }
}