// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

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
    public static Scalar<TNumber> Pow<TNumber>(this Scalar<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Pow(value, power);
    }

    /// <summary>
    /// Raises a <see cref="Vector{TNumber}"/> to the power of a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The base (<paramref name="value"/>) raised to the given <paramref name="power"/>.</returns>
    public static Vector<TNumber> Pow<TNumber>(this Vector<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Pow(value, power);
    }

    /// <summary>
    /// Raises a <see cref="Matrix{TNumber}"/> to the power of a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The base (<paramref name="value"/>) raised to the given <paramref name="power"/>.</returns>
    public static Matrix<TNumber> Pow<TNumber>(this Matrix<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Pow(value, power);
    }

    /// <summary>
    /// Raises a <see cref="Tensor{TNumber}"/> to the power of a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The base (<paramref name="value"/>) raised to the given <paramref name="power"/>.</returns>
    public static Tensor<TNumber> Pow<TNumber>(this Tensor<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Pow(value, power);
    }

    /// <summary>
    /// Raises a <see cref="Tensor{TNumber}"/> to the power of a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="value">The base.</param>
    /// <param name="power">The exponent.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>s.</typeparam>
    /// <returns>The base (<paramref name="value"/>) raised to the given <paramref name="power"/>.</returns>
    public static ITensor<TNumber> Pow<TNumber>(this ITensor<TNumber> value, Scalar<TNumber> power)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        if (value.IsScalar())
        {
            return TensorServiceProvider
                .GetTensorOperationServiceProvider()
                .GetPowerService()
                .Pow(value.AsScalar(), power);
        }

        if (value.IsVector())
        {
            return TensorServiceProvider
                .GetTensorOperationServiceProvider()
                .GetPowerService()
                .Pow(value.AsVector(), power);
        }

        if (value.IsMatrix())
        {
            return TensorServiceProvider
                .GetTensorOperationServiceProvider()
                .GetPowerService()
                .Pow(value.AsMatrix(), power);
        }

        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Pow(value.AsTensor(), power);
    }

    /// <summary>
    /// Raises a <see cref="Scalar{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The square of the <paramref name="value"/>.</returns>
    public static Scalar<TNumber> Square<TNumber>(this Scalar<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Square(value);
    }

    /// <summary>
    /// Raises a <see cref="Vector{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The square of the <paramref name="value"/>.</returns>
    public static Vector<TNumber> Square<TNumber>(this Vector<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Square(value);
    }

    /// <summary>
    /// Raises a <see cref="Matrix{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The square of the <paramref name="value"/>.</returns>
    public static Matrix<TNumber> Square<TNumber>(this Matrix<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Square(value);
    }

    /// <summary>
    /// Raises a <see cref="Tensor{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The square of the <paramref name="value"/>.</returns>
    public static Tensor<TNumber> Square<TNumber>(this Tensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Square(value);
    }

    /// <summary>
    /// Raises a <see cref="ITensor{TNumber}"/> to the power of 2.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The square of the <paramref name="value"/>.</returns>
    public static ITensor<TNumber> Square<TNumber>(this ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (value.IsScalar())
        {
            return TensorServiceProvider
                .GetTensorOperationServiceProvider()
                .GetPowerService()
                .Square(value.AsScalar());
        }

        if (value.IsVector())
        {
            return TensorServiceProvider
                .GetTensorOperationServiceProvider()
                .GetPowerService()
                .Square(value.AsVector());
        }

        if (value.IsMatrix())
        {
            return TensorServiceProvider
                .GetTensorOperationServiceProvider()
                .GetPowerService()
                .Square(value.AsMatrix());
        }

        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPowerService()
            .Square(value.AsTensor());
    }
}