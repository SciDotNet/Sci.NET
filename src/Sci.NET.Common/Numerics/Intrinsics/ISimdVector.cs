// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;

namespace Sci.NET.Common.Numerics.Intrinsics;

/// <summary>
/// A SIMD backend.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="ISimdVector{TNumber}"/>.</typeparam>
[PublicAPI]
[SuppressMessage("Design", "CA1000:Do not declare static members on generic types", Justification = "Reviewed.")]
public interface ISimdVector<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Gets the element at the provided index.
    /// </summary>
    /// <param name="index">The index.</param>
    public TNumber this[int index] { get; }

    /// <inheritdoc cref="Vector.Add{T}"/>
    public ISimdVector<TNumber> Add(ISimdVector<TNumber> other);

    /// <inheritdoc cref="Vector.Subtract{T}"/>
    public ISimdVector<TNumber> Subtract(ISimdVector<TNumber> other);

    /// <inheritdoc cref="Vector.Multiply{T}(System.Numerics.Vector{T},System.Numerics.Vector{T})"/>
    public ISimdVector<TNumber> Multiply(ISimdVector<TNumber> other);

    /// <inheritdoc cref="Vector.Divide{T}(System.Numerics.Vector{T},System.Numerics.Vector{T})"/>
    public ISimdVector<TNumber> Divide(ISimdVector<TNumber> other);

    /// <inheritdoc cref="Vector.SquareRoot{T}"/>
    public ISimdVector<TNumber> Sqrt();

    /// <inheritdoc cref="Vector.Abs{T}"/>
    public ISimdVector<TNumber> Abs();

    /// <inheritdoc cref="Vector.Negate{T}"/>
    public ISimdVector<TNumber> Negate();

    /// <inheritdoc cref="Vector.Max{T}"/>
    public ISimdVector<TNumber> Max(ISimdVector<TNumber> other);

    /// <inheritdoc cref="Vector.Min{T}"/>
    public ISimdVector<TNumber> Min(ISimdVector<TNumber> other);

    /// <summary>
    /// Clamps the vector between the provided min and max.
    /// </summary>
    /// <param name="min">The minimum value.</param>
    /// <param name="max">The maximum value.</param>
    /// <returns>The clamped vector.</returns>
    public ISimdVector<TNumber> Clamp(ISimdVector<TNumber> min, ISimdVector<TNumber> max);

    /// <inheritdoc cref="Vector.Dot{T}"/>
    public TNumber Dot(ISimdVector<TNumber> other);

    /// <inheritdoc cref="Vector.Sum{T}"/>
    public TNumber Sum();

    /// <summary>
    /// Copies the vector to the provided span.
    /// </summary>
    /// <param name="span">The destination span.</param>
    public void CopyTo(Span<TNumber> span);
}