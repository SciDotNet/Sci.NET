// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Backends;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// Represents a scalar.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="Scalar{TNumber}"/>.</typeparam>
[PublicAPI]
public class Scalar<TNumber> : Tensor<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Scalar{TNumber}"/> class.
    /// </summary>
    /// <param name="backend">The backend to use to store the <see cref="Scalar{TNumber}"/>.</param>
    public Scalar(ITensorBackend? backend = null)
        : base(backend)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Scalar{TNumber}"/> class.
    /// </summary>
    /// <param name="value">The value of the <see cref="Scalar{TNumber}"/>.</param>
    /// <param name="backend">The backend to use to store the <see cref="Scalar{TNumber}"/>.</param>
    public Scalar(TNumber value, ITensorBackend? backend = null)
        : base(backend)
    {
        Handle.Fill(value);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Scalar{TNumber}"/> class.
    /// </summary>
    /// <param name="handle">The handle for the memory block.</param>
    /// <param name="backend">The backend instance that was used.</param>
    internal Scalar(IMemoryBlock<TNumber> handle, ITensorBackend backend)
        : base(handle, Shape.Scalar, backend)
    {
    }

#pragma warning disable CS1591
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Scalar<TNumber> operator +(Scalar<TNumber> left, Scalar<TNumber> right)
    {
        return left.Add(right);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Scalar<TNumber> operator -(Scalar<TNumber> left, Scalar<TNumber> right)
    {
        return left.Subtract(right);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Scalar<TNumber> operator *(Scalar<TNumber> left, Scalar<TNumber> right)
    {
        return left.Multiply(right);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Scalar<TNumber> operator /(Scalar<TNumber> left, Scalar<TNumber> right)
    {
        return left.Divide(right);
    }
#pragma warning restore CS1591

    /// <summary>
    /// Gets the value of the <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <returns>The value of the <see cref="Scalar{TNumber}"/>.</returns>
    public TNumber GetValue()
    {
        var value = Handle.ToArray();
        return value[0];
    }
}