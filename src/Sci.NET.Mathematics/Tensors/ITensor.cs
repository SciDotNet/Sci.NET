// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Memory.ReferenceCounting;
using Sci.NET.Mathematics.Tensors.Backends;
using Sci.NET.Mathematics.Tensors.Elementwise;
using Sci.NET.Mathematics.Tensors.LinearAlgebra;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// An interface for a tensor, which is an immutable N-Dimensional array.
/// </summary>
/// <typeparam name="TNumber">The type of number stored by the tensor.</typeparam>
[PublicAPI]
public interface ITensor<TNumber> : IDisposable, IReferenceCounted
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Gets the memory block storing the tensor data.
    /// </summary>
    [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
    public IMemoryBlock<TNumber> Data { get; }

    /// <inheritdoc cref="Shape.Dimensions" />
    public int[] Dimensions { get; }

    /// <inheritdoc cref="Shape.Rank" />
    public int Rank { get; }

    /// <inheritdoc cref="Shape.ElementCount" />
    public long ElementCount { get; }

    /// <inheritdoc cref="Shape.Strides" />
    public long[] Strides { get; }

    /// <inheritdoc cref="Shape.IsScalar"/>
    public bool IsScalar { get; }

    /// <inheritdoc cref="Shape.IsVector"/>
    public bool IsVector { get; }

    /// <inheritdoc cref="Shape.IsMatrix"/>
    public bool IsMatrix { get; }

    /// <summary>
    /// Finds the sum of the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The sum of the two operands.</returns>
    public static ITensor<TNumber> operator +(ITensor<TNumber> left, ITensor<TNumber> right)
    {
        return TensorBackend.Instance.Arithmetic.Add(left, right);
    }

    /// <summary>
    /// Finds the sum of the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The sum of the two operands.</returns>
    public static ITensor<TNumber> operator +(TNumber left, ITensor<TNumber> right)
    {
        return TensorBackend.Instance.Arithmetic.Add(left, right);
    }

    /// <summary>
    /// Finds the difference between the right and left operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The difference between the right and left operands.</returns>
    public static ITensor<TNumber> operator -(ITensor<TNumber> left, ITensor<TNumber> right)
    {
        return TensorBackend.Instance.Arithmetic.Subtract(left, right);
    }

    /// <summary>
    /// Finds the difference between the right and left operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The difference between the left and right operands.</returns>
    public static ITensor<TNumber> operator -(TNumber left, ITensor<TNumber> right)
    {
        return TensorBackend.Instance.Arithmetic.Subtract(left, right);
    }

    /// <summary>
    /// Negates the values in the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to negate.</param>
    /// <returns>The element-wise negation of the input <see cref="ITensor{TNumber}"/>.</returns>
    public static ITensor<TNumber> operator -(ITensor<TNumber> tensor)
    {
        return TensorBackend.Instance.Arithmetic.Negate(tensor);
    }

    /// <summary>
    /// Finds the product of the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The product of the left and right operands.</returns>
    public static ITensor<TNumber> operator *(ITensor<TNumber> left, ITensor<TNumber> right)
    {
        return left.Dot(right);
    }

    /// <summary>
    /// Finds the product of the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The product of the left and right operands.</returns>
    public static ITensor<TNumber> operator *(TNumber left, ITensor<TNumber> right)
    {
        return right.ScalarProduct(left);
    }

    /// <summary>
    /// Finds the product of the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The product of the left and right operands.</returns>
    public static ITensor<TNumber> operator *(ITensor<TNumber> left, TNumber right)
    {
        return left.ScalarProduct(right);
    }

    /// <summary>
    /// Finds the quotient of the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The quotient of the left and right operands.</returns>
    public static ITensor<TNumber> operator /(ITensor<TNumber> left, ITensor<TNumber> right)
    {
        return TensorBackend.Instance.Arithmetic.Divide(left, right);
    }

    /// <summary>
    /// Finds the quotient of the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The quotient of the left and right operands.</returns>
    public static ITensor<TNumber> operator /(TNumber left, ITensor<TNumber> right)
    {
        return TensorBackend.Instance.Arithmetic.Divide(left, right);
    }

    /// <summary>
    /// Finds the quotient of the left and right operands.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The quotient of the left and right operands.</returns>
    public static ITensor<TNumber> operator /(ITensor<TNumber> left, TNumber right)
    {
        return TensorBackend.Instance.Arithmetic.Divide(left, right);
    }

    /// <summary>
    /// Gets the shape of the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <returns>The shape of the <see cref="ITensor{TNumber}"/>.</returns>
    public Shape GetShape();
}