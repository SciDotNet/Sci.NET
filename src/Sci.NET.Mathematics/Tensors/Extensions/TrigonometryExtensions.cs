﻿// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;

// ReSharper disable once CheckNamespace
#pragma warning disable IDE0130
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// A class containing extension methods for trigonometric functions.
/// </summary>
[PublicAPI]
public static class TrigonometryExtensions
{
    /// <summary>
    /// Computes the Sine of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Sine of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Sin<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sin(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Sine of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Sine of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Sin<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sin(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Sine of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Sine of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Sin<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sin(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Sine of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Sine of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Sin<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sin(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Sine of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Sine of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Sin<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sin(tensor)
            ;
    }

    /// <summary>
    /// Computes the Cosine of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Cosine of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Cos<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cos(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Cosine of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Cosine of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Cos<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cos(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Cosine of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Cosine of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Cos<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cos(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Cosine of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Cosine of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Cos<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cos(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Cosine of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Cosine of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Cos<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cos(tensor)
            ;
    }

    /// <summary>
    /// Computes the Tangent of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Tangent of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Tan<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tan(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Tangent of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Tangent of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Tan<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tan(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Tangent of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Tangent of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Tan<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tan(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Tangent of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Tangent of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Tan<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tan(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Tangent of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Tangent of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Tan<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tan(tensor)
            ;
    }

    /// <summary>
    /// Computes the Sine Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Sine Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Sin2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sin2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Sine Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Sine Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Sin2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sin2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Sine Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Sine Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Sin2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sin2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Sine Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Sine Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Sin2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sin2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Sine Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Sine Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Sin2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sin2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Cosine Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Cosine Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Cos2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cos2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Cosine Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Cosine Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Cos2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cos2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Cosine Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Cosine Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Cos2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cos2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Cosine Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Cosine Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Cos2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cos2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Cosine Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Cosine Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Cos2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cos2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Tangent Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Tangent Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Tan2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tan2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Tangent Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Tangent Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Tan2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tan2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Tangent Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Tangent Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Tan2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tan2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Tangent Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Tangent Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Tan2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tan2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Tangent Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Tangent Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Tan2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tan2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Sine of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Sine of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Sinh<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sinh(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Sine of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Sine of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Sinh<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sinh(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Sine of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Sine of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Sinh<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sinh(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Sine of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Sine of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Sinh<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sinh(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Sine of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Sine of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Sinh<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sinh(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Cosine of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosine of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Cosh<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cosh(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Cosine of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosine of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Cosh<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cosh(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Cosine of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosine of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Cosh<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cosh(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Cosine of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosine of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Cosh<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cosh(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Cosine of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosine of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Cosh<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cosh(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Tangent of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Tangent of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Tanh<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tanh(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Tangent of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Tangent of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Tanh<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tanh(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Tangent of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Tangent of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Tanh<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tanh(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Tangent of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Tangent of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Tanh<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tanh(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Tangent of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Tangent of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Tanh<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tanh(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Sine Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Sine Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Sinh2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sinh2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Sine Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Sine Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Sinh2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sinh2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Sine Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Sine Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Sinh2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sinh2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Sine Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Sine Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Sinh2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sinh2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Sine Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Sine Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Sinh2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sinh2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Cosine Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosine Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Cosh2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cosh2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Cosine Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosine Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Cosh2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cosh2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Cosine Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosine Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Cosh2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cosh2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Cosine Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosine Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Cosh2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cosh2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Cosine Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosine Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Cosh2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cosh2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Tangent Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Tangent Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Tanh2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tanh2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Tangent Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Tangent Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Tanh2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tanh2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Tangent Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Tangent Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Tanh2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tanh2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Tangent Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Tangent Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Tanh2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tanh2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Tangent Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Tangent Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Tanh2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tanh2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Inverse Sine of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Sine of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Asin<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Asin(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Inverse Sine of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Sine of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Asin<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Asin(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Inverse Sine of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Sine of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Asin<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Asin(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Inverse Sine of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Sine of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Asin<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Asin(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Inverse Sine of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Sine of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Asin<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Asin(tensor)
            ;
    }

    /// <summary>
    /// Computes the Inverse Cosine of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosine of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Acos<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acos(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Inverse Cosine of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosine of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Acos<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acos(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Inverse Cosine of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosine of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Acos<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acos(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Inverse Cosine of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosine of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Acos<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acos(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Inverse Cosine of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosine of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Acos<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acos(tensor)
            ;
    }

    /// <summary>
    /// Computes the Inverse Tangent of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Tangent of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Atan<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Atan(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Inverse Tangent of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Tangent of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Atan<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Atan(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Inverse Tangent of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Tangent of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Atan<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Atan(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Inverse Tangent of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Tangent of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Atan<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Atan(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Inverse Tangent of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Tangent of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Atan<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Atan(tensor)
            ;
    }

    /// <summary>
    /// Computes the Inverse Sine Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Sine Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Asin2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Asin2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Inverse Sine Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Sine Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Asin2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Asin2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Inverse Sine Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Sine Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Asin2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Asin2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Inverse Sine Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Sine Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Asin2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Asin2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Inverse Sine Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Sine Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Asin2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Asin2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Inverse Cosine Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosine Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Acos2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acos2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Inverse Cosine Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosine Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Acos2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acos2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Inverse Cosine Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosine Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Acos2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acos2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Inverse Cosine Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosine Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Acos2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acos2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Inverse Cosine Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosine Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Acos2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acos2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Inverse Tangent Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Tangent Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Atan2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Atan2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Inverse Tangent Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Tangent Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Atan2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Atan2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Inverse Tangent Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Tangent Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Atan2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Atan2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Inverse Tangent Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Tangent Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Atan2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Atan2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Inverse Tangent Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Tangent Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Atan2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Atan2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Sine of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Sine of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> ASinh<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASinh(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Sine of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Sine of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> ASinh<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASinh(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Sine of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Sine of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> ASinh<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASinh(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Sine of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Sine of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> ASinh<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASinh(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Sine of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Sine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Sine of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> ASinh<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASinh(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosine of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosine of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> ACosh<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACosh(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosine of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosine of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> ACosh<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACosh(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosine of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosine of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> ACosh<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACosh(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosine of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosine of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> ACosh<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACosh(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosine of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosine of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosine of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> ACosh<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACosh(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Tangent of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Tangent of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> ATanh<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ATanh(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Tangent of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Tangent of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> ATanh<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ATanh(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Tangent of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Tangent of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> ATanh<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ATanh(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Tangent of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Tangent of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> ATanh<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ATanh(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Tangent of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Tangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Tangent of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> ATanh<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ATanh(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Sine Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Sine Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> ASinh2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASinh2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Sine Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Sine Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> ASinh2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASinh2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Sine Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Sine Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> ASinh2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASinh2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Sine Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Sine Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> ASinh2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASinh2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Sine Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Sine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Sine Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> ASinh2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASinh2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosine Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosine Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> ACosh2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACosh2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosine Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosine Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> ACosh2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACosh2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosine Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosine Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> ACosh2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACosh2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosine Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosine Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> ACosh2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACosh2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosine Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosine Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosine Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> ACosh2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACosh2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Tangent Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Tangent Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> ATanh2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ATanh2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Tangent Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Tangent Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> ATanh2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ATanh2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Tangent Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Tangent Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> ATanh2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ATanh2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Tangent Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Tangent Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> ATanh2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ATanh2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Tangent Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Tangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Tangent Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> ATanh2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ATanh2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Cosecant of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Cosecant of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Csc<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csc(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Cosecant of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Cosecant of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Csc<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csc(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Cosecant of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Cosecant of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Csc<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csc(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Cosecant of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Cosecant of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Csc<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csc(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Cosecant of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Cosecant of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Csc<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csc(tensor)
            ;
    }

    /// <summary>
    /// Computes the Secant of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Secant of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Sec<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sec(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Secant of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Secant of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Sec<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sec(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Secant of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Secant of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Sec<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sec(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Secant of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Secant of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Sec<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sec(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Secant of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Secant of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Sec<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sec(tensor)
            ;
    }

    /// <summary>
    /// Computes the Cotangent of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Cotangent of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Cot<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cot(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Cotangent of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Cotangent of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Cot<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cot(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Cotangent of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Cotangent of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Cot<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cot(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Cotangent of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Cotangent of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Cot<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cot(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Cotangent of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Cotangent of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Cot<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cot(tensor)
            ;
    }

    /// <summary>
    /// Computes the Cosecant Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Cosecant Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Csc2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csc2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Cosecant Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Cosecant Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Csc2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csc2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Cosecant Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Cosecant Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Csc2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csc2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Cosecant Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Cosecant Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Csc2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csc2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Cosecant Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Cosecant Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Csc2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csc2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Secant Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Secant Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Sec2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sec2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Secant Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Secant Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Sec2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sec2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Secant Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Secant Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Sec2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sec2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Secant Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Secant Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Sec2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sec2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Secant Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Secant Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Sec2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sec2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Cotangent Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Cotangent Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Cot2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cot2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Cotangent Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Cotangent Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Cot2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cot2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Cotangent Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Cotangent Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Cot2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cot2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Cotangent Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Cotangent Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Cot2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cot2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Cotangent Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Cotangent Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Cot2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cot2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Cosecant of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosecant of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Csch<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csch(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Cosecant of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosecant of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Csch<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csch(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Cosecant of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosecant of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Csch<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csch(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Cosecant of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosecant of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Csch<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csch(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Cosecant of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosecant of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Csch<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csch(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Secant of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Secant of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Sech<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sech(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Secant of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Secant of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Sech<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sech(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Secant of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Secant of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Sech<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sech(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Secant of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Secant of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Sech<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sech(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Secant of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Secant of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Sech<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sech(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Cotangent of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cotangent of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Coth<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Coth(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Cotangent of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cotangent of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Coth<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Coth(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Cotangent of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cotangent of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Coth<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Coth(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Cotangent of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cotangent of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Coth<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Coth(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Cotangent of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cotangent of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Coth<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Coth(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Cosecant Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosecant Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Csch2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csch2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Cosecant Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosecant Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Csch2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csch2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Cosecant Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosecant Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Csch2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csch2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Cosecant Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosecant Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Csch2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csch2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Cosecant Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cosecant Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Csch2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Csch2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Secant Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Secant Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Sech2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sech2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Secant Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Secant Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Sech2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sech2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Secant Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Secant Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Sech2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sech2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Secant Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Secant Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Sech2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sech2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Secant Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Secant Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Sech2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sech2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Cotangent Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cotangent Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Coth2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Coth2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Cotangent Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cotangent Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Coth2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Coth2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Cotangent Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cotangent Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Coth2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Coth2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Cotangent Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cotangent Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Coth2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Coth2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Cotangent Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Cotangent Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Coth2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Coth2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Inverse Cosecant of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosecant of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Acsc<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acsc(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Inverse Cosecant of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosecant of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Acsc<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acsc(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Inverse Cosecant of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosecant of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Acsc<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acsc(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Inverse Cosecant of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosecant of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Acsc<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acsc(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Inverse Cosecant of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosecant of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Acsc<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acsc(tensor)
            ;
    }

    /// <summary>
    /// Computes the Inverse Secant of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Secant of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Asec<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Asec(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Inverse Secant of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Secant of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Asec<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Asec(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Inverse Secant of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Secant of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Asec<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Asec(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Inverse Secant of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Secant of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Asec<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Asec(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Inverse Secant of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Secant of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Asec<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Asec(tensor)
            ;
    }

    /// <summary>
    /// Computes the Inverse Cotangent of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cotangent of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Acot<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acot(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Inverse Cotangent of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cotangent of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Acot<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acot(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Inverse Cotangent of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cotangent of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Acot<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acot(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Inverse Cotangent of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cotangent of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Acot<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acot(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Inverse Cotangent of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cotangent of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Acot<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Acot(tensor)
            ;
    }

    /// <summary>
    /// Computes the Inverse Cosecant Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosecant Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> ACsc2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACsc2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Inverse Cosecant Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosecant Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> ACsc2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACsc2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Inverse Cosecant Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosecant Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> ACsc2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACsc2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Inverse Cosecant Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosecant Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> ACsc2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACsc2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Inverse Cosecant Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cosecant Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> ACsc2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACsc2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Inverse Secant Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Secant Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> ASec2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASec2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Inverse Secant Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Secant Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> ASec2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASec2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Inverse Secant Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Secant Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> ASec2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASec2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Inverse Secant Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Secant Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> ASec2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASec2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Inverse Secant Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Secant Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> ASec2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASec2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Inverse Cotangent Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cotangent Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> ACot2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACot2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Inverse Cotangent Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cotangent Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> ACot2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACot2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Inverse Cotangent Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cotangent Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> ACot2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACot2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Inverse Cotangent Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cotangent Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> ACot2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACot2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Inverse Cotangent Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Inverse Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Inverse Cotangent Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> ACot2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACot2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosecant of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosecant of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> ACsch<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACsch(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosecant of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosecant of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> ACsch<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACsch(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosecant of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosecant of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> ACsch<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACsch(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosecant of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosecant of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> ACsch<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACsch(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosecant of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosecant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosecant of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> ACsch<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACsch(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Secant of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Secant of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> ASech<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASech(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Secant of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Secant of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> ASech<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASech(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Secant of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Secant of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> ASech<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASech(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Secant of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Secant of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> ASech<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASech(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Secant of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Secant of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Secant of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> ASech<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASech(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cotangent of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cotangent of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> ACoth<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACoth(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cotangent of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cotangent of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> ACoth<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACoth(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cotangent of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cotangent of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> ACoth<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACoth(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cotangent of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cotangent of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> ACoth<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACoth(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cotangent of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cotangent of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cotangent of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> ACoth<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACoth(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosecant Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosecant Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> ACsch2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACsch2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosecant Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosecant Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> ACsch2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACsch2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosecant Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosecant Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> ACsch2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACsch2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosecant Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosecant Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> ACsch2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACsch2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cosecant Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cosecant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cosecant Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> ACsch2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACsch2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Secant Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Secant Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> ASech2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASech2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Secant Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Secant Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> ASech2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASech2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Secant Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Secant Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> ASech2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASech2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Secant Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Secant Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> ASech2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASech2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Secant Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Secant Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Secant Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> ASech2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ASech2(tensor)
            ;
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cotangent Squared of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cotangent Squared of the specified <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> ACoth2<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACoth2(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cotangent Squared of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cotangent Squared of the specified <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> ACoth2<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACoth2(tensor)
            .ToVector();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cotangent Squared of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cotangent Squared of the specified <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> ACoth2<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACoth2(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cotangent Squared of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cotangent Squared of the specified <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> ACoth2<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACoth2(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Computes the Hyperbolic Inverse Cotangent Squared of the specified <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The tensor to calculate the Hyperbolic Inverse Cotangent Squared of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The Hyperbolic Inverse Cotangent Squared of the specified <see cref="ITensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> ACoth2<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .ACoth2(tensor)
            ;
    }

#pragma warning disable RCS1036

#pragma warning restore RCS1036
}
