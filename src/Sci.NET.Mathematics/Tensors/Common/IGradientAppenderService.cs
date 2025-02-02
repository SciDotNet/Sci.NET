// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;

namespace Sci.NET.Mathematics.Tensors.Common;

/// <summary>
/// A service for adding gradients to tensors.
/// </summary>
[PublicAPI]
public interface IGradientAppenderService
{
    /// <summary>
    /// Adds the gradient of the <paramref name="result"/> <see cref="ITensor{TNumber}"/> with respect to the <paramref name="left"/> and <paramref name="right"/> <see cref="ITensor{TNumber}"/>s if required.
    /// </summary>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="left">The left <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="overrideRequiresGradient">Overrides the <see cref="ITensor{TNumber}.RequiresGradient"/> flag of the <paramref name="result"/> <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="leftGradientFunction">The function to compute the gradient of the <paramref name="result"/> <see cref="ITensor{TNumber}"/> with respect to the <paramref name="left"/> <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="rightGradientFunction">The function to compute the gradient of the <paramref name="result"/> <see cref="ITensor{TNumber}"/> with respect to the <paramref name="right"/> <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="name">The name of the operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    public void AddGradientIfRequired<TNumber>(
        ref ITensor<TNumber> result,
        ITensor<TNumber> left,
        ITensor<TNumber> right,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> leftGradientFunction,
        Func<ITensor<TNumber>, ITensor<TNumber>> rightGradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Adds the gradient of the <paramref name="result"/> <see cref="Scalar{TNumber}"/> with respect to the <paramref name="left"/> and <paramref name="right"/> <see cref="ITensor{TNumber}"/>s if required.
    /// </summary>
    /// <param name="result">The result <see cref="Scalar{TNumber}"/>.</param>
    /// <param name="left">The left <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="overrideRequiresGradient">Overrides the <see cref="ITensor{TNumber}.RequiresGradient"/> flag of the <paramref name="result"/> <see cref="Scalar{TNumber}"/>.</param>
    /// <param name="leftGradientFunction">The function to compute the gradient of the <paramref name="result"/> <see cref="Scalar{TNumber}"/> with respect to the <paramref name="left"/> <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="rightGradientFunction">The function to compute the gradient of the <paramref name="result"/> <see cref="Scalar{TNumber}"/> with respect to the <paramref name="right"/> <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="name">The name of the operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    public void AddGradientIfRequired<TNumber>(
        ref Scalar<TNumber> result,
        ITensor<TNumber> left,
        ITensor<TNumber> right,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> leftGradientFunction,
        Func<ITensor<TNumber>, ITensor<TNumber>> rightGradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Adds the gradient of the <paramref name="result"/> <see cref="Vector{TNumber}"/> with respect to the <paramref name="left"/> and <paramref name="right"/> <see cref="ITensor{TNumber}"/>s if required.
    /// </summary>
    /// <param name="result">The result <see cref="Vector{TNumber}"/>.</param>
    /// <param name="left">The left <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="overrideRequiresGradient">Overrides the <see cref="ITensor{TNumber}.RequiresGradient"/> flag of the <paramref name="result"/> <see cref="Vector{TNumber}"/>.</param>
    /// <param name="leftGradientFunction">The function to compute the gradient of the <paramref name="result"/> <see cref="Vector{TNumber}"/> with respect to the <paramref name="left"/> <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="rightGradientFunction">The function to compute the gradient of the <paramref name="result"/> <see cref="Vector{TNumber}"/> with respect to the <paramref name="right"/> <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="name">The name of the operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    public void AddGradientIfRequired<TNumber>(
        ref Vector<TNumber> result,
        ITensor<TNumber> left,
        ITensor<TNumber> right,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> leftGradientFunction,
        Func<ITensor<TNumber>, ITensor<TNumber>> rightGradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Adds the gradient of the <paramref name="result"/> <see cref="Matrix{TNumber}"/> with respect to the <paramref name="left"/> and <paramref name="right"/> <see cref="ITensor{TNumber}"/>s if required.
    /// </summary>
    /// <param name="result">The result <see cref="Matrix{TNumber}"/>.</param>
    /// <param name="left">The left <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="overrideRequiresGradient">Overrides the <see cref="ITensor{TNumber}.RequiresGradient"/> flag of the <paramref name="result"/> <see cref="Matrix{TNumber}"/>.</param>
    /// <param name="leftGradientFunction">The function to compute the gradient of the <paramref name="result"/> <see cref="Matrix{TNumber}"/> with respect to the <paramref name="left"/> <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="rightGradientFunction">The function to compute the gradient of the <paramref name="result"/> <see cref="Matrix{TNumber}"/> with respect to the <paramref name="right"/> <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="name">The name of the operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    public void AddGradientIfRequired<TNumber>(
        ref Matrix<TNumber> result,
        ITensor<TNumber> left,
        ITensor<TNumber> right,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> leftGradientFunction,
        Func<ITensor<TNumber>, ITensor<TNumber>> rightGradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Adds the gradient of the <paramref name="result"/> <see cref="Tensor{TNumber}"/> with respect to the <paramref name="left"/> and <paramref name="right"/> <see cref="ITensor{TNumber}"/>s if required.
    /// </summary>
    /// <param name="result">The result <see cref="Tensor{TNumber}"/>.</param>
    /// <param name="left">The left <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="right">The right <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="overrideRequiresGradient">Overrides the <see cref="ITensor{TNumber}.RequiresGradient"/> flag of the <paramref name="result"/> <see cref="Tensor{TNumber}"/>.</param>
    /// <param name="leftGradientFunction">The function to compute the gradient of the <paramref name="result"/> <see cref="Tensor{TNumber}"/> with respect to the <paramref name="left"/> <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="rightGradientFunction">The function to compute the gradient of the <paramref name="result"/> <see cref="Tensor{TNumber}"/> with respect to the <paramref name="right"/> <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="name">The name of the operation.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    public void AddGradientIfRequired<TNumber>(
        ref Tensor<TNumber> result,
        ITensor<TNumber> left,
        ITensor<TNumber> right,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> leftGradientFunction,
        Func<ITensor<TNumber>, ITensor<TNumber>> rightGradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Adds the gradient of the <paramref name="result"/> <see cref="ITensor{TNumber}"/> with respect to the <paramref name="input"/> <see cref="ITensor{TNumber}"/> if required.
    /// </summary>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="input">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="overrideRequiresGradient">Overrides the <see cref="ITensor{TNumber}.RequiresGradient"/> flag of the <paramref name="result"/> <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="gradientFunction">The function to compute the gradient of the <paramref name="result"/> <see cref="ITensor{TNumber}"/> with respect to the <paramref name="input"/> <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="name">The name of the operation.</param>
    /// <typeparam name="TNumber">The number type of the tensors.</typeparam>
    public void AddGradientIfRequired<TNumber>(
        ref ITensor<TNumber> result,
        ITensor<TNumber> input,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> gradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Adds the gradient of the <paramref name="result"/> <see cref="Scalar{TNumber}"/> with respect to the <paramref name="input"/> <see cref="ITensor{TNumber}"/> if required.
    /// </summary>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="input">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="overrideRequiresGradient">Overrides the <see cref="ITensor{TNumber}.RequiresGradient"/> flag of the <paramref name="result"/> <see cref="Scalar{TNumber}"/>.</param>
    /// <param name="gradientFunction">The function to compute the gradient of the <paramref name="result"/> <see cref="Scalar{TNumber}"/> with respect to the <paramref name="input"/> <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="name">The name of the operation.</param>
    /// <typeparam name="TNumber">The number type of the tensors.</typeparam>
    public void AddGradientIfRequired<TNumber>(
        ref Scalar<TNumber> result,
        ITensor<TNumber> input,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> gradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Adds the gradient of the <paramref name="result"/> <see cref="Vector{TNumber}"/> with respect to the <paramref name="input"/> <see cref="ITensor{TNumber}"/> if required.
    /// </summary>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="input">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="overrideRequiresGradient">Overrides the <see cref="ITensor{TNumber}.RequiresGradient"/> flag of the <paramref name="result"/> <see cref="Vector{TNumber}"/>.</param>
    /// <param name="gradientFunction">The function to compute the gradient of the <paramref name="result"/> <see cref="Vector{TNumber}"/> with respect to the <paramref name="input"/> <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="name">The name of the operation.</param>
    /// <typeparam name="TNumber">The number type of the tensors.</typeparam>
    public void AddGradientIfRequired<TNumber>(
        ref Vector<TNumber> result,
        ITensor<TNumber> input,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> gradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Adds the gradient of the <paramref name="result"/> <see cref="Matrix{TNumber}"/> with respect to the <paramref name="input"/> <see cref="ITensor{TNumber}"/> if required.
    /// </summary>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="input">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="overrideRequiresGradient">Overrides the <see cref="ITensor{TNumber}.RequiresGradient"/> flag of the <paramref name="result"/> <see cref="Matrix{TNumber}"/>.</param>
    /// <param name="gradientFunction">The function to compute the gradient of the <paramref name="result"/> <see cref="Matrix{TNumber}"/> with respect to the <paramref name="input"/> <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="name">The name of the operation.</param>
    /// <typeparam name="TNumber">The number type of the tensors.</typeparam>
    public void AddGradientIfRequired<TNumber>(
        ref Matrix<TNumber> result,
        ITensor<TNumber> input,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> gradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Adds the gradient of the <paramref name="result"/> <see cref="Tensor{TNumber}"/> with respect to the <paramref name="input"/> <see cref="ITensor{TNumber}"/> if required.
    /// </summary>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="input">The input <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="overrideRequiresGradient">Overrides the <see cref="ITensor{TNumber}.RequiresGradient"/> flag of the <paramref name="result"/> <see cref="Tensor{TNumber}"/>.</param>
    /// <param name="gradientFunction">The function to compute the gradient of the <paramref name="result"/> <see cref="Tensor{TNumber}"/> with respect to the <paramref name="input"/> <see cref="ITensor{TNumber}"/>.</param>
    /// <param name="name">The name of the operation.</param>
    /// <typeparam name="TNumber">The number type of the tensors.</typeparam>
    public void AddGradientIfRequired<TNumber>(
        ref Tensor<TNumber> result,
        ITensor<TNumber> input,
        bool? overrideRequiresGradient,
        Func<ITensor<TNumber>, ITensor<TNumber>> gradientFunction,
        [CallerMemberName] string? name = null)
        where TNumber : unmanaged, INumber<TNumber>;
}