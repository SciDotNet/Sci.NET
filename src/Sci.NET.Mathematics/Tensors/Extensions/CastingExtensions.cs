// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;

#pragma warning disable IDE0130

// ReSharper disable once CheckNamespace
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// Provides casting extensions for <see cref="ITensor{TNumber}"/>.
/// </summary>
[PublicAPI]
public static class CastingExtensions
{
    /// <summary>
    /// Casts an <see cref="ITensor{TNumber}"/> to a new type.
    /// </summary>
    /// <param name="tensor">The tensor to cast.</param>
    /// <typeparam name="TIn">The input number type.</typeparam>
    /// <typeparam name="TOut">The output number type.</typeparam>
    /// <returns>The input cast to <typeparamref name="TOut"/>.</returns>
    public static ITensor<TOut> Cast<TIn, TOut>(this ITensor<TIn> tensor)
        where TIn : unmanaged, System.Numerics.INumber<TIn>
        where TOut : unmanaged, System.Numerics.INumber<TOut>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetCastingService()
            .Cast<TIn, TOut>(tensor.ToTensor());
    }

    /// <summary>
    /// Casts a <see cref="Scalar{TNumber}"/> to a new type.
    /// </summary>
    /// <param name="input">The <see cref="Scalar{TNumber}"/> to cast.</param>
    /// <typeparam name="TIn">The input number type.</typeparam>
    /// <typeparam name="TOut">The output number type.</typeparam>
    /// <returns>The input cast to <typeparamref name="TOut"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TOut> Cast<TIn, TOut>(this Scalar<TIn> input)
        where TIn : unmanaged, System.Numerics.INumber<TIn>
        where TOut : unmanaged, System.Numerics.INumber<TOut>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetCastingService()
            .Cast<TIn, TOut>(input);
    }

    /// <summary>
    /// Casts a <see cref="Vector{TNumber}"/> to a new type.
    /// </summary>
    /// <param name="input">The <see cref="Vector{TNumber}"/> to cast.</param>
    /// <typeparam name="TIn">The input number type.</typeparam>
    /// <typeparam name="TOut">The output number type.</typeparam>
    /// <returns>The input cast to <typeparamref name="TOut"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TOut> Cast<TIn, TOut>(this Vector<TIn> input)
        where TIn : unmanaged, System.Numerics.INumber<TIn>
        where TOut : unmanaged, System.Numerics.INumber<TOut>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetCastingService()
            .Cast<TIn, TOut>(input);
    }

    /// <summary>
    /// Casts a <see cref="Matrix{TNumber}"/> to a new type.
    /// </summary>
    /// <param name="input">The <see cref="Matrix{TNumber}"/> to cast.</param>
    /// <typeparam name="TIn">The input number type.</typeparam>
    /// <typeparam name="TOut">The output number type.</typeparam>
    /// <returns>The input cast to <typeparamref name="TOut"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TOut> Cast<TIn, TOut>(this Matrix<TIn> input)
        where TIn : unmanaged, System.Numerics.INumber<TIn>
        where TOut : unmanaged, System.Numerics.INumber<TOut>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetCastingService()
            .Cast<TIn, TOut>(input);
    }

    /// <summary>
    /// Casts a <see cref="Tensor{TNumber}"/> to a new type.
    /// </summary>
    /// <param name="input">The <see cref="Tensor{TNumber}"/> to cast.</param>
    /// <typeparam name="TIn">The input number type.</typeparam>
    /// <typeparam name="TOut">The output number type.</typeparam>
    /// <returns>The input cast to <typeparamref name="TOut"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TOut> Cast<TIn, TOut>(this Tensor<TIn> input)
        where TIn : unmanaged, System.Numerics.INumber<TIn>
        where TOut : unmanaged, System.Numerics.INumber<TOut>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetCastingService()
            .Cast<TIn, TOut>(input);
    }
}