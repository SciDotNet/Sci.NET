// Copyright (c) Sci.NET Foundation. All rights reserved.
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
    /// Calculates the sin of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="scalar">The <see cref="Scalar{TNumber}"/> to calculate the sin of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The sin of the <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Sin<TNumber>(this Scalar<TNumber> scalar)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sin(scalar);
    }

    /// <summary>
    /// Calculates the sin of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="vector">The <see cref="Vector{TNumber}"/> to calculate the sin of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The sin of the <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Sin<TNumber>(this Vector<TNumber> vector)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sin(vector);
    }

    /// <summary>
    /// Calculates the sin of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="matrix">The <see cref="Matrix{TNumber}"/> to calculate the sin of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The sin of the <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Sin<TNumber>(this Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sin(matrix);
    }

    /// <summary>
    /// Calculates the sin of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Tensor{TNumber}"/> to calculate the sin of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The sin of the <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Sin<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Sin(tensor);
    }

    /// <summary>
    /// Calculates the cosine of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="scalar">The <see cref="Scalar{TNumber}"/> to calculate the cos of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The cosine of the <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Cos<TNumber>(this Scalar<TNumber> scalar)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cos(scalar);
    }

    /// <summary>
    /// Calculates the cosine of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="vector">The <see cref="Vector{TNumber}"/> to calculate the cos of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The cosine of the <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Cos<TNumber>(this Vector<TNumber> vector)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cos(vector);
    }

    /// <summary>
    /// Calculates the cosine of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="matrix">The <see cref="Matrix{TNumber}"/> to calculate the cos of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The cosine of the <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Cos<TNumber>(this Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cos(matrix);
    }

    /// <summary>
    /// Calculates the cosine of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Tensor{TNumber}"/> to calculate the cos of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The cosine of the <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Cos<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Cos(tensor);
    }

    /// <summary>
    /// Calculates the tan of the specified <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="scalar">The <see cref="Scalar{TNumber}"/> to calculate the tan of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The tan of the <see cref="Scalar{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Tan<TNumber>(this Scalar<TNumber> scalar)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tan(scalar);
    }

    /// <summary>
    /// Calculates the tan of the specified <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="vector">The <see cref="Vector{TNumber}"/> to calculate the tan of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The tan of the <see cref="Vector{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Tan<TNumber>(this Vector<TNumber> vector)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tan(vector);
    }

    /// <summary>
    /// Calculates the tan of the specified <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="matrix">The <see cref="Matrix{TNumber}"/> to calculate the tan of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The tan of the <see cref="Matrix{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Tan<TNumber>(this Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tan(matrix);
    }

    /// <summary>
    /// Calculates the tan of the specified <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Tensor{TNumber}"/> to calculate the tan of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The tan of the <see cref="Tensor{TNumber}"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Tan<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetTrigonometryService()
            .Tan(tensor);
    }
}