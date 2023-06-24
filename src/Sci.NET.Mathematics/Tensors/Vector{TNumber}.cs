// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Backends;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// Represents a vector.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="Vector{TNumber}"/>.</typeparam>
[PublicAPI]
public class Vector<TNumber> : Tensor<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Vector{TNumber}"/> class.
    /// </summary>
    /// <param name="length">The length of the vector.</param>
    /// <param name="backend">The backend instance to use.</param>
    public Vector(int length, ITensorBackend? backend = null)
        : base(backend, length)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Vector{TNumber}"/> class.
    /// </summary>
    /// <param name="length">The length of the <see cref="Vector{TNumber}"/>.</param>
    /// <param name="memoryBlock">The memory block of the <see cref="Vector{TNumber}"/>.</param>
    /// <param name="tensorBackend">The backend instance.</param>
    public Vector(int length, IMemoryBlock<TNumber> memoryBlock, ITensorBackend tensorBackend)
        : base(memoryBlock, new Shape(length), tensorBackend)
    {
    }

    /// <summary>
    /// Gets the length of the vector.
    /// </summary>
    public int Length => Shape[0];

    /// <summary>
    /// Gets the debugger display object.
    /// </summary>
    [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
    private protected Array DebuggerDisplayObject => ToArray();

#pragma warning disable CS1591
    public static Tensor<TNumber> operator +(Vector<TNumber> left, Scalar<TNumber> right)
    {
        return left.Add(right);
    }

    public static Tensor<TNumber> operator +(Scalar<TNumber> left, Vector<TNumber> right)
    {
        return left.Add(right);
    }

    public static Tensor<TNumber> operator +(Vector<TNumber> left, Vector<TNumber> right)
    {
        return left.Add(right);
    }

    public static Tensor<TNumber> operator -(Vector<TNumber> left, Scalar<TNumber> right)
    {
        return left.Subtract(right);
    }

    public static Tensor<TNumber> operator -(Scalar<TNumber> left, Vector<TNumber> right)
    {
        return left.Subtract(right);
    }

    public static Tensor<TNumber> operator -(Vector<TNumber> left, Vector<TNumber> right)
    {
        return left.Subtract(right);
    }

    public static Tensor<TNumber> operator *(Vector<TNumber> left, Scalar<TNumber> right)
    {
        return right.Multiply(left);
    }

    public static Tensor<TNumber> operator *(Scalar<TNumber> left, Vector<TNumber> right)
    {
        return right.Multiply(left);
    }
#pragma warning restore CS1591
}