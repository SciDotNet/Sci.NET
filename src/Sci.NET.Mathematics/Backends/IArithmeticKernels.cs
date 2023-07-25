// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Attributes;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for arithmetic backends.
/// </summary>
[PublicAPI]
public interface IArithmeticKernels
{
    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Add<TNumber>(
        [AssumesShape("1")] Scalar<TNumber> left,
        [AssumesShape("1")] Scalar<TNumber> right,
        [AssumesShape("1")] Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Add<TNumber>(
        [AssumesShape("1")] Scalar<TNumber> left,
        [AssumesShape("i")] Tensors.Vector<TNumber> right,
        [AssumesShape("i")] Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Add<TNumber>(
        [AssumesShape("1")] Scalar<TNumber> left,
        [AssumesShape("i,j")] Matrix<TNumber> right,
        [AssumesShape("i,j")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Add<TNumber>(
        [AssumesShape("1")] Scalar<TNumber> left,
        [AssumesShape("*")] Tensor<TNumber> right,
        [AssumesShape(nameof(right))] Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Add<TNumber>(
        [AssumesShape("i")] Tensors.Vector<TNumber> left,
        [AssumesShape("1")] Scalar<TNumber> right,
        [AssumesShape("i")] Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Add<TNumber>(
        [AssumesShape("i")] Tensors.Vector<TNumber> left,
        [AssumesShape("i")] Tensors.Vector<TNumber> right,
        [AssumesShape("i")] Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Add<TNumber>(
        [AssumesShape("i")] Tensors.Vector<TNumber> left,
        [AssumesShape("i,j")] Matrix<TNumber> right,
        [AssumesShape("i, j")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Add<TNumber>(
        [AssumesShape("i,j")] Matrix<TNumber> left,
        [AssumesShape("1")] Scalar<TNumber> right,
        [AssumesShape("i,j")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Add<TNumber>(
        [AssumesShape("i,j")] Matrix<TNumber> left,
        [AssumesShape("i")] Tensors.Vector<TNumber> right,
        [AssumesShape("i,j")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Add<TNumber>(
        [AssumesShape("i,j")] Matrix<TNumber> left,
        [AssumesShape("i,j")] Matrix<TNumber> right,
        [AssumesShape("i,j")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Add<TNumber>(
        [AssumesShape("*")] Tensor<TNumber> left,
        [AssumesShape("1")] Scalar<TNumber> right,
        [AssumesShape(nameof(left))] Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the sum of the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Add<TNumber>(
        [AssumesShape("*")] Tensor<TNumber> left,
        [AssumesShape(nameof(left))] Tensor<TNumber> right,
        [AssumesShape(nameof(left))] Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Subtract<TNumber>(
        [AssumesShape("1")] Scalar<TNumber> left,
        [AssumesShape("1")] Scalar<TNumber> right,
        [AssumesShape("1")] Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Subtract<TNumber>(
        [AssumesShape("1")] Scalar<TNumber> left,
        [AssumesShape("i")] Tensors.Vector<TNumber> right,
        [AssumesShape("i")] Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Subtract<TNumber>(
        [AssumesShape("1")] Scalar<TNumber> left,
        [AssumesShape("i,j")] Matrix<TNumber> right,
        [AssumesShape("i,j")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="left"/> <see cref="Scalar{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Subtract<TNumber>(
        [AssumesShape("1")] Scalar<TNumber> left,
        [AssumesShape("*")] Tensor<TNumber> right,
        [AssumesShape(nameof(right))] Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Subtract<TNumber>(
        [AssumesShape("i")] Tensors.Vector<TNumber> left,
        [AssumesShape("1")] Scalar<TNumber> right,
        [AssumesShape("i")] Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Subtract<TNumber>(
        [AssumesShape("i")] Tensors.Vector<TNumber> left,
        [AssumesShape("i")] Tensors.Vector<TNumber> right,
        [AssumesShape("i")] Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="left"/> <see cref="Tensors.Vector{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Subtract<TNumber>(
        [AssumesShape("i")] Tensors.Vector<TNumber> left,
        [AssumesShape("i,j")] Matrix<TNumber> right,
        [AssumesShape("i,j")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Subtract<TNumber>(
        [AssumesShape("i,j")] Matrix<TNumber> left,
        [AssumesShape("1")] Scalar<TNumber> right,
        [AssumesShape("i,j")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Tensors.Vector{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Subtract<TNumber>(
        [AssumesShape("i,j")] Matrix<TNumber> left,
        [AssumesShape("i")] Tensors.Vector<TNumber> right,
        [AssumesShape("i,j")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="left"/> <see cref="Matrix{TNumber}"/> and <paramref name="right"/> <see cref="Matrix{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Subtract<TNumber>(
        [AssumesShape("i,j")] Matrix<TNumber> left,
        [AssumesShape("i,j")] Matrix<TNumber> right,
        [AssumesShape("i,j")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Scalar{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Subtract<TNumber>(
        [AssumesShape("*")] Tensor<TNumber> left,
        [AssumesShape("1")] Scalar<TNumber> right,
        [AssumesShape(nameof(left))] Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the difference between the <paramref name="left"/> <see cref="Tensor{TNumber}"/> and <paramref name="right"/> <see cref="Tensor{TNumber}"/>,
    /// and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The result <see cref="ITensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the operands and result.</typeparam>
    [AssumesValidDevice]
    public void Subtract<TNumber>(
        [AssumesShape("*")] Tensor<TNumber> left,
        [AssumesShape(nameof(left))] Tensor<TNumber> right,
        [AssumesShape(nameof(left))] Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a scalar multiplication of a <see cref="Scalar{TNumber}"/> and a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The product of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Multiply<TNumber>(
        [AssumesShape("1")] Scalar<TNumber> left,
        [AssumesShape("1")] Scalar<TNumber> right,
        [AssumesShape("1")] Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a scalar multiplication of a <see cref="Scalar{TNumber}"/> and a <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The product of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Multiply<TNumber>(
        [AssumesShape("1")] Scalar<TNumber> left,
        [AssumesShape("i")] Tensors.Vector<TNumber> right,
        [AssumesShape("i")] Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a scalar multiplication of a <see cref="Scalar{TNumber}"/> and a <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The product of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Multiply<TNumber>(
        [AssumesShape("1")] Scalar<TNumber> left,
        [AssumesShape("i,j")] Matrix<TNumber> right,
        [AssumesShape("i,j")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a scalar multiplication of a <see cref="Scalar{TNumber}"/> and a <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The product of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Multiply<TNumber>(
        [AssumesShape("1")] Scalar<TNumber> left,
        [AssumesShape("*")] Tensor<TNumber> right,
        [AssumesShape(nameof(right))] Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a scalar multiplication of a <see cref="Tensors.Vector{TNumber}"/> and a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The product of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Multiply<TNumber>(
        [AssumesShape("i")] Tensors.Vector<TNumber> left,
        [AssumesShape("1")] Scalar<TNumber> right,
        [AssumesShape("i")] Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a point-wise multiplication of a <see cref="Tensors.Vector{TNumber}"/> and a <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">A <see cref="Tensors.Vector{TNumber}"/> containing the point-wise product of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensors.Vector{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Multiply<TNumber>(
        [AssumesShape("i")] Tensors.Vector<TNumber> left,
        [AssumesShape("i")] Tensors.Vector<TNumber> right,
        [AssumesShape("i")] Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the element-wise product of the <paramref name="left"/> <see cref="ITensor{TNumber}"/> and <paramref name="right"/> <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">Stores the element-wise product of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Multiply<TNumber>(
        [AssumesShape("i")] Tensors.Vector<TNumber> left,
        [AssumesShape("i,j")] Matrix<TNumber> right,
        [AssumesShape("i,j")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a scalar multiplication of a <see cref="Matrix{TNumber}"/> and a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The product of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Multiply<TNumber>(
        [AssumesShape("i,j")] Matrix<TNumber> left,
        [AssumesShape("1")] Scalar<TNumber> right,
        [AssumesShape("i,j")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a point-wise multiplication of a <see cref="Matrix{TNumber}"/> and a <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The product of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/> and <see cref="Tensors.Vector{TNumber}"/> instances.</typeparam>
    public void Multiply<TNumber>(
        [AssumesShape("i,j")] Matrix<TNumber> left,
        [AssumesShape("i,j")] Matrix<TNumber> right,
        [AssumesShape("i,j")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a scalar multiplication of a <see cref="Scalar{TNumber}"/> and a <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The product of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Multiply<TNumber>(
        [AssumesShape("*")] Tensor<TNumber> left,
        [AssumesShape("1")] Scalar<TNumber> right,
        [AssumesShape(nameof(left))] Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a point-wise multiplication of a <see cref="Tensor{TNumber}"/> and a <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The product of the <paramref name="left"/> and <paramref name="right"/> operands.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Multiply<TNumber>(
        [AssumesShape("*")] Tensor<TNumber> left,
        [AssumesShape(nameof(left))] Tensor<TNumber> right,
        [AssumesShape(nameof(left))] Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a scalar multiplication of a <see cref="Scalar{TNumber}"/> and a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The product of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Divide<TNumber>(
        [AssumesShape("1")] Scalar<TNumber> left,
        [AssumesShape("1")] Scalar<TNumber> right,
        [AssumesShape("1")] Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a scalar multiplication of a <see cref="Scalar{TNumber}"/> and a <see cref="Tensors.Vector{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The product of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Divide<TNumber>(
        [AssumesShape("1")] Scalar<TNumber> left,
        [AssumesShape("i")] Tensors.Vector<TNumber> right,
        [AssumesShape("i")] Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a scalar multiplication of a <see cref="Scalar{TNumber}"/> and a <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The product of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Divide<TNumber>(
        [AssumesShape("1")] Scalar<TNumber> left,
        [AssumesShape("i,j")] Matrix<TNumber> right,
        [AssumesShape("i,j")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a scalar multiplication of a <see cref="Scalar{TNumber}"/> and a <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The product of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Divide<TNumber>(
        [AssumesShape("1")] Scalar<TNumber> left,
        [AssumesShape("*")] Tensor<TNumber> right,
        [AssumesShape(nameof(right))] Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a scalar multiplication of a <see cref="Tensors.Vector{TNumber}"/> and a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The product of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Divide<TNumber>(
        [AssumesShape("*")] Tensors.Vector<TNumber> left,
        [AssumesShape("1")] Scalar<TNumber> right,
        [AssumesShape("*")] Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a scalar multiplication of a <see cref="Matrix{TNumber}"/> and a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The product of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Divide<TNumber>(
        [AssumesShape("*")] Matrix<TNumber> left,
        [AssumesShape("1")] Scalar<TNumber> right,
        [AssumesShape("*")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Performs a scalar multiplication of a <see cref="Tensor{TNumber}"/> and a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">The product of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Divide<TNumber>(
        [AssumesShape("*")] Tensor<TNumber> left,
        [AssumesShape("1")] Scalar<TNumber> right,
        [AssumesShape("*")] Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Negates the <paramref name="value"/> <see cref="Scalar{TNumber}"/> and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="value">The value to negate.</param>
    /// <param name="result">The result <see cref="Scalar{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    [AssumesValidDevice]
    public void Negate<TNumber>(
        [AssumesShape("1")] Scalar<TNumber> value,
        [AssumesShape("1")] Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Negates the <paramref name="value"/> <see cref="Tensors.Vector{TNumber}"/> and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="value">The value to negate.</param>
    /// <param name="result">The result <see cref="Tensors.Vector{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensors.Vector{TNumber}"/>.</typeparam>
    [AssumesValidDevice]
    public void Negate<TNumber>(
        [AssumesShape("i")] Tensors.Vector<TNumber> value,
        [AssumesShape("i")] Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Negates the <paramref name="value"/> <see cref="Matrix{TNumber}"/> and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="value">The value to negate.</param>
    /// <param name="result">The result <see cref="Matrix{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    [AssumesValidDevice]
    public void Negate<TNumber>(
        [AssumesShape("i,j")] Matrix<TNumber> value,
        [AssumesShape("i,j")] Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Negates the <paramref name="value"/> <see cref="Tensor{TNumber}"/> and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="value">The value to negate.</param>
    /// <param name="result">The result <see cref="Tensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    [AssumesValidDevice]
    public void Negate<TNumber>(
        [AssumesShape("*")] Tensor<TNumber> value,
        [AssumesShape(nameof(value))] Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the absolute value of the <paramref name="value"/> <see cref="Scalar{TNumber}"/> and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="value">The value to find the absolute value of.</param>
    /// <param name="result">The result <see cref="Scalar{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
    [AssumesValidDevice]
    public void Abs<TNumber>(Scalar<TNumber> value, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the absolute value of the <paramref name="value"/> <see cref="Tensors.Vector{TNumber}"/> and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="value">The value to find the absolute value of.</param>
    /// <param name="result">The result <see cref="Tensors.Vector{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensors.Vector{TNumber}"/>.</typeparam>
    [AssumesValidDevice]
    public void Abs<TNumber>(Tensors.Vector<TNumber> value, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the absolute value of the <paramref name="value"/> <see cref="Matrix{TNumber}"/> and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="value">The value to find the absolute value of.</param>
    /// <param name="result">The result <see cref="Matrix{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    [AssumesValidDevice]
    public void Abs<TNumber>(Matrix<TNumber> value, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the absolute value of the <paramref name="value"/> <see cref="Tensor{TNumber}"/> and stores the result in the <paramref name="result"/> parameter.
    /// </summary>
    /// <param name="value">The value to find the absolute value of.</param>
    /// <param name="result">The result <see cref="Tensor{TNumber}"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    [AssumesValidDevice]
    public void Abs<TNumber>(Tensor<TNumber> value, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Finds the element-wise square root of the <paramref name="tensor"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to find the square root of.</param>
    /// <param name="result">Stores the element-wise square root of the <paramref name="tensor"/>.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Sqrt<TNumber>(ITensor<TNumber> tensor, Tensor<TNumber> result)
        where TNumber : unmanaged, IRootFunctions<TNumber>, INumber<TNumber>;

    /// <summary>
    /// Finds the element-wise quotient of the <paramref name="left"/> <see cref="ITensor{TNumber}"/> and <paramref name="right"/> <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="result">Stores the element-wise quotient of the <paramref name="left"/> and <paramref name="right"/> parameters.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/> instances.</typeparam>
    [AssumesValidDevice]
    public void Divide<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>;
}