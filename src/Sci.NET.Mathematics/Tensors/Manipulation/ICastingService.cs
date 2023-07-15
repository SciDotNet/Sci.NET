// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Manipulation;

/// <summary>
/// Provides tensor casting functionality.
/// </summary>
[PublicAPI]
public interface ICastingService
{
    /// <summary>
    /// Casts a <see cref="Scalar{TNumber}"/> to a different type.
    /// </summary>
    /// <param name="input">The <see cref="Scalar{TNumber}"/> to cast.</param>
    /// <typeparam name="TIn">The number type of the input <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <typeparam name="TOut">The number type of the output <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The input <see cref="Scalar{TNumber}"/> cast to <typeparamref name="TOut"/>.</returns>
    public Scalar<TOut> Cast<TIn, TOut>(Scalar<TIn> input)
        where TIn : unmanaged, INumber<TIn>
        where TOut : unmanaged, INumber<TOut>;

    /// <summary>
    /// Casts a <see cref="Vector{TNumber}"/> to a different type.
    /// </summary>
    /// <param name="input">The <see cref="Vector{TNumber}"/> to cast.</param>
    /// <typeparam name="TIn">The number type of the input <see cref="Vector{TNumber}"/>.</typeparam>
    /// <typeparam name="TOut">The number type of the output <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The input <see cref="Vector{TNumber}"/> cast to <typeparamref name="TOut"/>.</returns>
    public Vector<TOut> Cast<TIn, TOut>(Vector<TIn> input)
        where TIn : unmanaged, INumber<TIn>
        where TOut : unmanaged, INumber<TOut>;

    /// <summary>
    /// Casts a <see cref="Matrix{TNumber}"/> to a different type.
    /// </summary>
    /// <param name="input">The <see cref="Matrix{TNumber}"/> to cast.</param>
    /// <typeparam name="TIn">The number type of the input <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <typeparam name="TOut">The number type of the output <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The input <see cref="Matrix{TNumber}"/> cast to <typeparamref name="TOut"/>.</returns>
    public Matrix<TOut> Cast<TIn, TOut>(Matrix<TIn> input)
        where TIn : unmanaged, INumber<TIn>
        where TOut : unmanaged, INumber<TOut>;

    /// <summary>
    /// Casts a <see cref="Tensor{TNumber}"/> to a different type.
    /// </summary>
    /// <param name="input">The <see cref="Tensor{TNumber}"/> to cast.</param>
    /// <typeparam name="TIn">The number type of the input <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <typeparam name="TOut">The number type of the output <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The input <see cref="Matrix{TNumber}"/> cast to <typeparamref name="TOut"/>.</returns>
    public Tensor<TOut> Cast<TIn, TOut>(Tensor<TIn> input)
        where TIn : unmanaged, INumber<TIn>
        where TOut : unmanaged, INumber<TOut>;
}