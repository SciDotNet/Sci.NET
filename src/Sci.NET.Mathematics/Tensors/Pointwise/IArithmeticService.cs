// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Pointwise;

/// <summary>
/// An interface providing methods for <see cref="ITensor{TNumber}"/> arithmetic operations.
/// </summary>
[PublicAPI]
public interface IArithmeticService
{
    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Scalar<TNumber> Add<TNumber>(
        Scalar<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Vector<TNumber> Add<TNumber>(
        Scalar<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Matrix<TNumber> Add<TNumber>(
        Scalar<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Add<TNumber>(
        Scalar<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Vector<TNumber> Add<TNumber>(
        Vector<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Vector<TNumber> Add<TNumber>(
        Vector<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Matrix<TNumber> Add<TNumber>(
        Vector<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>A new <see cref="Tensor{TNumber}"/> containing the sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Add<TNumber>(
        Vector<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Matrix<TNumber> Add<TNumber>(
        Matrix<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Matrix<TNumber> Add<TNumber>(
        Matrix<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Matrix<TNumber> Add<TNumber>(
        Matrix<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Add<TNumber>(
        Matrix<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Add<TNumber>(
        Tensor<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>A new <see cref="Tensor{TNumber}"/> containing the sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Add<TNumber>(
        Tensor<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>A new <see cref="Tensor{TNumber}"/> containing the sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Add<TNumber>(
        Tensor<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Add<TNumber>(
        Tensor<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="ITensor{TNumber}"/> and <paramref name="right"/> <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>A new <see cref="ITensor{TNumber}"/> containing the sum of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public ITensor<TNumber> Add<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Scalar{TNumber}"/> and <paramref name="left"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    public Scalar<TNumber> Subtract<TNumber>(
        Scalar<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Scalar{TNumber}"/> and <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    public Vector<TNumber> Subtract<TNumber>(
        Scalar<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Scalar{TNumber}"/> and <paramref name="left"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    public Matrix<TNumber> Subtract<TNumber>(
        Scalar<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Scalar{TNumber}"/> and <paramref name="left"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    public Tensor<TNumber> Subtract<TNumber>(
        Scalar<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="left"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    public Vector<TNumber> Subtract<TNumber>(
        Vector<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    public Vector<TNumber> Subtract<TNumber>(
        Vector<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="left"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    public Matrix<TNumber> Subtract<TNumber>(
        Vector<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Vector{TNumber}"/> and <paramref name="left"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The difference between the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    public Tensor<TNumber> Subtract<TNumber>(
        Vector<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Matrix{TNumber}"/> and <paramref name="left"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    public Matrix<TNumber> Subtract<TNumber>(
        Matrix<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Matrix{TNumber}"/> and <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    public Matrix<TNumber> Subtract<TNumber>(
        Matrix<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Matrix{TNumber}"/> and <paramref name="left"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    public Matrix<TNumber> Subtract<TNumber>(
        Matrix<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Matrix{TNumber}"/> and <paramref name="left"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The difference between the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    public Tensor<TNumber> Subtract<TNumber>(
        Matrix<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Tensor{TNumber}"/> and <paramref name="left"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    public Tensor<TNumber> Subtract<TNumber>(
        Tensor<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Tensor{TNumber}"/> and <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>A new <see cref="Tensor{TNumber}"/> containing the difference between the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    public Tensor<TNumber> Subtract<TNumber>(
        Tensor<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Tensor{TNumber}"/> and <paramref name="left"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The difference between the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    public Tensor<TNumber> Subtract<TNumber>(
        Tensor<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="Tensor{TNumber}"/> and <paramref name="left"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The sum of the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    public Tensor<TNumber> Subtract<TNumber>(
        Tensor<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="right"/> <see cref="ITensor{TNumber}"/> and <paramref name="left"/> <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The difference between the <paramref name="right"/> and <paramref name="left"/> operands.</returns>
    public ITensor<TNumber> Subtract<TNumber>(
        ITensor<TNumber> left,
        ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Scalar<TNumber> Multiply<TNumber>(
        Scalar<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Vector<TNumber> Multiply<TNumber>(
        Scalar<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Matrix<TNumber> Multiply<TNumber>(
        Scalar<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Multiply<TNumber>(
        Scalar<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Vector<TNumber> Multiply<TNumber>(
        Vector<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>A new <see cref="Vector{TNumber}"/> containing the product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Vector<TNumber> Multiply<TNumber>(
        Vector<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>A new <see cref="Matrix{TNumber}"/> containing the product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Matrix<TNumber> Multiply<TNumber>(
        Vector<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>A new <see cref="Tensor{TNumber}"/> containing the product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Multiply<TNumber>(
        Vector<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Matrix<TNumber> Multiply<TNumber>(
        Matrix<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>A new <see cref="Vector{TNumber}"/> containing the product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Matrix<TNumber> Multiply<TNumber>(
        Matrix<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands.</typeparam>
    /// <returns>A new <see cref="Matrix{TNumber}"/> containing the product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Matrix<TNumber> Multiply<TNumber>(
        Matrix<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The type of the operands and result.</typeparam>
    /// <returns>A new <see cref="Tensor{TNumber}"/> containing the product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Matrix<TNumber> Multiply<TNumber>(
        Matrix<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Multiply<TNumber>(
        Tensor<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The type of the operands and result.</typeparam>
    /// <returns>A new <see cref="Vector{TNumber}"/> containing the product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Multiply<TNumber>(
        Tensor<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The type of the operands and result.</typeparam>
    /// <returns>A new <see cref="Matrix{TNumber}"/> containing the product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Multiply<TNumber>(
        Tensor<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the product of the <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Multiply<TNumber>(
        Tensor<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Scalar<TNumber> Divide<TNumber>(
        Scalar<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Vector<TNumber> Divide<TNumber>(
        Scalar<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Matrix<TNumber> Divide<TNumber>(
        Scalar<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Divide<TNumber>(
        Scalar<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Vector{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Vector<TNumber> Divide<TNumber>(
        Vector<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Vector{TNumber}"/> and <paramref name="right"/> <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>A new <see cref="Vector{TNumber}"/> containing the quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Vector<TNumber> Divide<TNumber>(
        Vector<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Vector{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>A new <see cref="Matrix{TNumber}"/> containing the quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Matrix<TNumber> Divide<TNumber>(
        Vector<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Vector{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The type of the operands and result.</typeparam>
    /// <returns>A new <see cref="Tensor{TNumber}"/> containing the quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Divide<TNumber>(
        Vector<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Matrix<TNumber> Divide<TNumber>(
        Matrix<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Matrix<TNumber> Divide<TNumber>(
        Matrix<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Matrix<TNumber> Divide<TNumber>(
        Matrix<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Divide<TNumber>(
        Matrix<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Divide<TNumber>(
        Tensor<TNumber> left,
        Scalar<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Divide<TNumber>(
        Tensor<TNumber> left,
        Vector<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Divide<TNumber>(
        Tensor<TNumber> left,
        Matrix<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the quotient of the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The type of the operands and result.</typeparam>
    /// <returns>The quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public Tensor<TNumber> Divide<TNumber>(
        Tensor<TNumber> left,
        Tensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Negates the <paramref name="value"/>.
    /// </summary>
    /// <param name="value">The value to negate.</param>
    /// <typeparam name="TNumber">The number type of the <paramref name="value"/>.</typeparam>
    /// <returns>The negated <paramref name="value"/>.</returns>
    public Scalar<TNumber> Negate<TNumber>(Scalar<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Negates the <paramref name="value"/>.
    /// </summary>
    /// <param name="value">The value to negate.</param>
    /// <typeparam name="TNumber">The number type of the <paramref name="value"/>.</typeparam>
    /// <returns>The negated <paramref name="value"/>.</returns>
    public Vector<TNumber> Negate<TNumber>(Vector<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Negates the <paramref name="value"/>.
    /// </summary>
    /// <param name="value">The value to negate.</param>
    /// <typeparam name="TNumber">The number type of the <paramref name="value"/>.</typeparam>
    /// <returns>The negated <paramref name="value"/>.</returns>
    public Matrix<TNumber> Negate<TNumber>(Matrix<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Negates the <paramref name="value"/>.
    /// </summary>
    /// <param name="value">The value to negate.</param>
    /// <typeparam name="TNumber">The number type of the <paramref name="value"/>.</typeparam>
    /// <returns>The negated <paramref name="value"/>.</returns>
    public Tensor<TNumber> Negate<TNumber>(Tensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the absolute value of the <paramref name="value"/>.
    /// </summary>
    /// <param name="value">The value to find the absolute value of.</param>
    /// <typeparam name="TNumber">The number type of the <paramref name="value"/>.</typeparam>
    /// <returns>The absolute value of the <paramref name="value"/>.</returns>
    public Scalar<TNumber> Abs<TNumber>(Scalar<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the absolute value of the <paramref name="value"/>.
    /// </summary>
    /// <param name="value">The value to find the absolute value of.</param>
    /// <typeparam name="TNumber">The number type of the <paramref name="value"/>.</typeparam>
    /// <returns>The absolute value of the <paramref name="value"/>.</returns>
    public Vector<TNumber> Abs<TNumber>(Vector<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the absolute value of the <paramref name="value"/>.
    /// </summary>
    /// <param name="value">The value to find the absolute value of.</param>
    /// <typeparam name="TNumber">The number type of the <paramref name="value"/>.</typeparam>
    /// <returns>The absolute value of the <paramref name="value"/>.</returns>
    public Matrix<TNumber> Abs<TNumber>(Matrix<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the absolute value of the <paramref name="value"/>.
    /// </summary>
    /// <param name="value">The value to find the absolute value of.</param>
    /// <typeparam name="TNumber">The number type of the <paramref name="value"/>.</typeparam>
    /// <returns>The absolute value of the <paramref name="value"/>.</returns>
    public Tensor<TNumber> Abs<TNumber>(Tensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the element-wise product of the <paramref name="left"/> and <paramref name="right"/> <see cref="ITensor{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands.</typeparam>
    /// <returns>A new <see cref="ITensor{TNumber}"/> containing the element-wise product of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public ITensor<TNumber> Multiply<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the element-wise quotient of the <paramref name="left"/> and <paramref name="right"/> <see cref="ITensor{TNumber}"/>s.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the operands.</typeparam>
    /// <returns>A new <see cref="ITensor{TNumber}"/> containing the element-wise quotient of the <paramref name="left"/> and <paramref name="right"/> operands.</returns>
    public ITensor<TNumber> Divide<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Computes the square root of a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to compute the square root of.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The square root of the <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> Sqrt<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, IRootFunctions<TNumber>, INumber<TNumber>;
}