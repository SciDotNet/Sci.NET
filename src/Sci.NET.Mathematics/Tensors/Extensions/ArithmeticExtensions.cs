// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;

#pragma warning disable IDE0130

// ReSharper disable once CheckNamespace
namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// Extension methods for <see cref="ITensor{TNumber}"/> arithmetic operations.
/// </summary>
[PublicAPI]
public static class ArithmeticExtensions
{
    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Add<TNumber>(
        this Scalar<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Add(left, right);
    }

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Add<TNumber>(
        this
            Scalar<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Add(left, right);
    }

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Add<TNumber>(
        this
            Scalar<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Add(left, right);
    }

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Add<TNumber>(
        this
            Scalar<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Add(left, right);
    }

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Add<TNumber>(
        this
            Vector<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Add(left, right);
    }

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Add<TNumber>(
        this
            Vector<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Add(left, right);
    }

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Add<TNumber>(
        this
            Vector<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Add(left, right);
    }

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Add<TNumber>(
        this
            Matrix<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Add(left, right);
    }

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Add<TNumber>(
        this Matrix<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Add(left, right);
    }

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Add<TNumber>(
        this
            Matrix<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Add(left, right);
    }

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Add<TNumber>(
        this
            Tensor<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Add(left, right);
    }

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Add<TNumber>(
        this
            Tensor<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Add(left, right);
    }

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Scalar{TNumber}"/> and <paramref name="left"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Subtract<TNumber>(
        this
            Scalar<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Subtract(left, right);
    }

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Scalar{TNumber}"/> and <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Subtract<TNumber>(
        this
            Scalar<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Subtract(left, right);
    }

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Scalar{TNumber}"/> and <paramref name="left"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Subtract<TNumber>(
        this
            Scalar<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Subtract(left, right);
    }

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Scalar{TNumber}"/> and <paramref name="left"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Subtract<TNumber>(
        this
            Scalar<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Subtract(left, right);
    }

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="left"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Subtract<TNumber>(
        this
            Vector<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Subtract(left, right);
    }

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Subtract<TNumber>(
        this
            Vector<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Subtract(left, right);
    }

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="left"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Subtract<TNumber>(
        this
            Vector<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Subtract(left, right);
    }

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Matrix{TNumber}"/> and <paramref name="left"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Subtract<TNumber>(
        this
            Matrix<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Subtract(left, right);
    }

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Matrix{TNumber}"/> and <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Subtract<TNumber>(
        this
            Matrix<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Subtract(left, right);
    }

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Matrix{TNumber}"/> and <paramref name="left"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Subtract<TNumber>(
        this
            Matrix<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Subtract(left, right);
    }

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Tensor{TNumber}"/> and <paramref name="left"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Subtract<TNumber>(
        this
            Tensor<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Subtract(left, right);
    }

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Tensor{TNumber}"/> and <paramref name="left"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Subtract<TNumber>(
        this
            Tensor<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Subtract(left, right);
    }

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Multiply<TNumber>(this Scalar<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Multiply(left, right);
    }

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Multiply<TNumber>(this Scalar<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Multiply(left, right);
    }

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Multiply<TNumber>(this Scalar<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Multiply(left, right);
    }

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Multiply<TNumber>(this Scalar<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Multiply(left, right);
    }

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Vector{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Multiply<TNumber>(this Vector<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Multiply(left, right);
    }

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Multiply<TNumber>(this Matrix<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Multiply(left, right);
    }

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Multiply<TNumber>(this Tensor<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Multiply(left, right);
    }

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Vector{TNumber}"/> and <paramref name="right"/> <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public static Tensor<TNumber> Multiply<TNumber>(this Tensor<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Multiply(left, right);
    }

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Divide<TNumber>(this Scalar<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Divide(left, right);
    }

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Divide<TNumber>(this Scalar<TNumber> left, Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Divide(left, right);
    }

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Divide<TNumber>(this Scalar<TNumber> left, Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Divide(left, right);
    }

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Divide<TNumber>(this Scalar<TNumber> left, Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Divide(left, right);
    }

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Vector{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Divide<TNumber>(this Vector<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Divide(left, right);
    }

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Divide<TNumber>(this Matrix<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Divide(left, right);
    }

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Divide<TNumber>(this Tensor<TNumber> left, Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Divide(left, right);
    }

    /// <summary>
    /// Finds the negation of the <paramref name="scalar"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="scalar">The <see cref="Scalar{TNumber}"/> to negate.</param>
    /// <typeparam name="TNumber">The number type of the <paramref name="scalar"/> and result.</typeparam>
    /// <returns>The negation of the <paramref name="scalar"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Negate<TNumber>(this Scalar<TNumber> scalar)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Negate(scalar);
    }

    /// <summary>
    /// Finds the negation of the <paramref name="vector"/> <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="vector">The <see cref="Vector{TNumber}"/> to negate.</param>
    /// <typeparam name="TNumber">The number type of the <paramref name="vector"/> and result.</typeparam>
    /// <returns>The negation of the <paramref name="vector"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Negate<TNumber>(this Vector<TNumber> vector)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Negate(vector);
    }

    /// <summary>
    /// Finds the negation of the <paramref name="matrix"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="matrix">The <see cref="Matrix{TNumber}"/> to negate.</param>
    /// <typeparam name="TNumber">The number type of the <paramref name="matrix"/> and result.</typeparam>
    /// <returns>The negation of the <paramref name="matrix"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Negate<TNumber>(this Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Negate(matrix);
    }

    /// <summary>
    /// Finds the negation of the <paramref name="tensor"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Tensor{TNumber}"/> to negate.</param>
    /// <typeparam name="TNumber">The number type of the <paramref name="tensor"/> and result.</typeparam>
    /// <returns>The negation of the <paramref name="tensor"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Negate<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Negate(tensor);
    }

    /// <summary>
    /// Finds the absolute value of the <paramref name="scalar"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="scalar">The <see cref="Scalar{TNumber}"/> to find the absolute value of.</param>
    /// <typeparam name="TNumber">The number type of the <paramref name="scalar"/> and result.</typeparam>
    /// <returns>The absolute value of the <paramref name="scalar"/>.</returns>
    [DebuggerStepThrough]
    public static Scalar<TNumber> Abs<TNumber>(this Scalar<TNumber> scalar)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Abs(scalar);
    }

    /// <summary>
    /// Finds the absolute value of the <paramref name="vector"/> <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="vector">The <see cref="Vector{TNumber}"/> to find the absolute value of.</param>
    /// <typeparam name="TNumber">The number type of the <paramref name="vector"/> and result.</typeparam>
    /// <returns>The absolute value of the <paramref name="vector"/>.</returns>
    [DebuggerStepThrough]
    public static Vector<TNumber> Abs<TNumber>(this Vector<TNumber> vector)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Abs(vector);
    }

    /// <summary>
    /// Finds the absolute value of the <paramref name="matrix"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="matrix">The <see cref="Matrix{TNumber}"/> to find the absolute value of.</param>
    /// <typeparam name="TNumber">The number type of the <paramref name="matrix"/> and result.</typeparam>
    /// <returns>The absolute value of the <paramref name="matrix"/>.</returns>
    [DebuggerStepThrough]
    public static Matrix<TNumber> Abs<TNumber>(this Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Abs(matrix);
    }

    /// <summary>
    /// Finds the absolute value of the <paramref name="tensor"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Tensor{TNumber}"/> to find the absolute value of.</param>
    /// <typeparam name="TNumber">The number type of the <paramref name="tensor"/> and result.</typeparam>
    /// <returns>The absolute value of the <paramref name="tensor"/>.</returns>
    [DebuggerStepThrough]
    public static Tensor<TNumber> Abs<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetArithmeticService()
            .Abs(tensor);
    }
}

#pragma warning restore IDE0130