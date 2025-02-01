// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;

#pragma warning disable IDE0130

// ReSharper disable once CheckNamespace
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// Extensions for <see cref="ITensor{TNumber}"/> power operations.
/// </summary>
[PublicAPI]
public static class PowerExtensions
{
    /// <summary>
    /// Raises a <see cref="Scalar{TNumber}"/> to the power of a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The base (<paramref name="value"/>) raised to the given <paramref name="power"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Pow<TNumber>(this Scalar<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Pow(value, power)
            .ToScalar();
    }

    /// <summary>
    /// Raises a <see cref="Vector{TNumber}"/> to the power of a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The base (<paramref name="value"/>) raised to the given <paramref name="power"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Pow<TNumber>(this Vector<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Pow(value, power)
            .ToVector();
    }

    /// <summary>
    /// Raises a <see cref="Matrix{TNumber}"/> to the power of a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The base (<paramref name="value"/>) raised to the given <paramref name="power"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Pow<TNumber>(this Matrix<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Pow(value, power)
            .ToMatrix();
    }

    /// <summary>
    /// Raises a <see cref="Tensor{TNumber}"/> to the power of a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The base (<paramref name="value"/>) raised to the given <paramref name="power"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Pow<TNumber>(this Tensor<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Pow(value, power)
            .ToTensor();
    }

    /// <summary>
    /// Raises a <see cref="Tensor{TNumber}"/> to the power of a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The base (<paramref name="value"/>) raised to the given <paramref name="power"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Pow<TNumber>(this ITensor<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Pow(value, power);
    }

    /// <summary>
    /// Raises a <see cref="Scalar{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The square of the <paramref name="value"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Square<TNumber>(this Scalar<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Square(value)
            .ToScalar();
    }

    /// <summary>
    /// Raises a <see cref="Vector{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The square of the <paramref name="value"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Square<TNumber>(this Vector<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Square(value)
            .ToVector();
    }

    /// <summary>
    /// Raises a <see cref="Matrix{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The square of the <paramref name="value"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Square<TNumber>(this Matrix<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Square(value)
            .ToMatrix();
    }

    /// <summary>
    /// Raises a <see cref="Tensor{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The square of the <paramref name="value"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Square<TNumber>(this Tensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Square(value)
            .ToTensor();
    }

    /// <summary>
    /// Raises a <see cref="ITensor{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The square of the <paramref name="value"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Square<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Square(value.ToTensor());
    }

    /// <summary>
    /// Raises e to the power of <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="value">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The result of e raised to the given <paramref name="value"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Exp<TNumber>(this Scalar<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Exp(value)
            .ToScalar();
    }

    /// <summary>
    /// Raises e to the power of <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="value">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The result of e raised to the given <paramref name="value"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Exp<TNumber>(this Vector<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Exp(value)
            .ToVector();
    }

    /// <summary>
    /// Raises e to the power of <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="value">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The result of e raised to the given <paramref name="value"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Exp<TNumber>(this Matrix<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Exp(value)
            .ToMatrix();
    }

    /// <summary>
    /// Raises e to the power of <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The result of e raised to the given <paramref name="value"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Exp<TNumber>(this Tensor<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Exp(value)
            .ToTensor();
    }

    /// <summary>
    /// Raises e to the power of <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="value">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of e raised to the given <paramref name="value"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Exp<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Exp(value);
    }

    /// <summary>
    /// Finds the natural logarithm of a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Scalar{TNumber}"/> to find the natural logarithm of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The natural logarithm of the <paramref name="tensor"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Log<TNumber>(this Scalar<TNumber> tensor)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Log(tensor)
            .ToScalar();
    }

    /// <summary>
    /// Finds the natural logarithm of a <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Vector{TNumber}"/> to find the natural logarithm of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
    /// <returns>The natural logarithm of the <paramref name="tensor"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Log<TNumber>(this Vector<TNumber> tensor)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Log(tensor)
            .ToVector();
    }

    /// <summary>
    /// Finds the natural logarithm of a <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Matrix{TNumber}"/> to find the natural logarithm of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The natural logarithm of the <paramref name="tensor"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Log<TNumber>(this Matrix<TNumber> tensor)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Log(tensor)
            .ToMatrix();
    }

    /// <summary>
    /// Finds the natural logarithm of a <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Tensor{TNumber}"/> to find the natural logarithm of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The natural logarithm of the <paramref name="tensor"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Log<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Log(tensor)
            .ToTensor();
    }

    /// <summary>
    /// Finds the natural logarithm of a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Scalar{TNumber}"/> to find the natural logarithm of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    /// <returns>The natural logarithm of the <paramref name="tensor"/>.</returns>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Log<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Log(tensor);
    }
}