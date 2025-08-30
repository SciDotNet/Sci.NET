// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.Intrinsics;

namespace Sci.NET.Mathematics.Backends.Managed.BinaryOps;

/// <summary>
/// An interface for binary tensor operations that can be applied element-wise to tensors.
/// </summary>
internal interface IManagedBinaryTensorOp
{
    /// <summary>
    /// Determines if AVX intrinsics are supported for the given numeric type <typeparamref name="TNumber"/>.
    /// </summary>
    /// <typeparam name="TNumber">The numeric type of the tensor elements.</typeparam>
    /// <returns>><c>true</c> if AVX intrinsics are supported; otherwise, <c>false</c>.</returns>
    public static abstract bool IsAvxSupported<TNumber>();

    /// <summary>
    /// Invokes the binary operation on two elements of type <typeparamref name="TNumber"/>.
    /// </summary>
    /// <typeparam name="TNumber">The numeric type of the tensor elements.</typeparam>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the binary operation.</returns>
    public static abstract TNumber Invoke<TNumber>(TNumber left, TNumber right)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Invokes the binary operation on two <see cref="float"/> values.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The result of the binary operation.</returns>
    public static abstract float Invoke(float left, float right);

    /// <summary>
    /// Invokes the binary operation on two <see cref="double"/> values.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">>The right operand.</param>
    /// <returns>The result of the binary operation.</returns>
    public static abstract double Invoke(double left, double right);

    /// <summary>
    /// Invokes the binary operation using AVX intrinsics on two <see cref="Vector256{T}"/> of type <see cref="float"/>.
    /// </summary>
    /// <param name="left">The left operand vector.</param>
    /// <param name="right">The right operand vector.</param>
    /// <returns>The result vector of the binary operation.</returns>
    public static abstract Vector256<float> InvokeAvx(Vector256<float> left, Vector256<float> right);

    /// <summary>
    /// Invokes the binary operation using AVX intrinsics on two <see cref="Vector256{T}"/> of type <see cref="double"/>.
    /// </summary>
    /// <param name="left">The left operand vector.</param>
    /// <param name="right">The right operand vector.</param>
    /// <returns>The result vector of the binary operation.</returns>
    public static abstract Vector256<double> InvokeAvx(Vector256<double> left, Vector256<double> right);
}