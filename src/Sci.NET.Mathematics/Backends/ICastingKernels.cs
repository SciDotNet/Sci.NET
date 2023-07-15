// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for casting kernels.
/// </summary>
[PublicAPI]
public interface ICastingKernels
{
    /// <summary>
    /// Casts a <see cref="Scalar{TNumber}"/> to a different type.
    /// </summary>
    /// <param name="input">The <see cref="Scalar{TNumber}"/> to cast.</param>
    /// <param name="output">The result <see cref="Scalar{TNumber}"/>.</param>
    /// <typeparam name="TIn">The number type of the input <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <typeparam name="TOut">The number type of the output <see cref="Scalar{TNumber}"/>.</typeparam>
    public void Cast<TIn, TOut>(Scalar<TIn> input, Scalar<TOut> output)
        where TIn : unmanaged, System.Numerics.INumber<TIn>
        where TOut : unmanaged, System.Numerics.INumber<TOut>;

    /// <summary>
    /// Casts a <see cref="Vector{TNumber}"/> to a different type.
    /// </summary>
    /// <param name="input">The <see cref="Vector{TNumber}"/> to cast.</param>
    /// <param name="output">The result <see cref="Vector{TNumber}"/>.</param>
    /// <typeparam name="TIn">The number type of the input <see cref="Vector{TNumber}"/>.</typeparam>
    /// <typeparam name="TOut">The number type of the output <see cref="Vector{TNumber}"/>.</typeparam>
    public void Cast<TIn, TOut>(Vector<TIn> input, Vector<TOut> output)
        where TIn : unmanaged, System.Numerics.INumber<TIn>
        where TOut : unmanaged, System.Numerics.INumber<TOut>;

    /// <summary>
    /// Casts a <see cref="Matrix{TNumber}"/> to a different type.
    /// </summary>
    /// <param name="input">The <see cref="Matrix{TNumber}"/> to cast.</param>
    /// <param name="output">The result <see cref="Matrix{TNumber}"/>.</param>
    /// <typeparam name="TIn">The number type of the input <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <typeparam name="TOut">The number type of the output <see cref="Matrix{TNumber}"/>.</typeparam>
    public void Cast<TIn, TOut>(Matrix<TIn> input, Matrix<TOut> output)
        where TIn : unmanaged, System.Numerics.INumber<TIn>
        where TOut : unmanaged, System.Numerics.INumber<TOut>;

    /// <summary>
    /// Casts a <see cref="Tensor{TNumber}"/> to a different type.
    /// </summary>
    /// <param name="input">The <see cref="Tensor{TNumber}"/> to cast.</param>
    /// <param name="output">The result <see cref="Tensor{TNumber}"/>.</param>
    /// <typeparam name="TIn">The number type of the input <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <typeparam name="TOut">The number type of the output <see cref="Tensor{TNumber}"/>.</typeparam>
    public void Cast<TIn, TOut>(Tensor<TIn> input, Tensor<TOut> output)
        where TIn : unmanaged, System.Numerics.INumber<TIn>
        where TOut : unmanaged, System.Numerics.INumber<TOut>;
}