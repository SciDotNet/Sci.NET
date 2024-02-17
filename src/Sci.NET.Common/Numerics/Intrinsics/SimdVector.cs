// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Performance;

namespace Sci.NET.Common.Numerics.Intrinsics;

/// <summary>
/// A helper class for SIMD vectors.
/// </summary>
[PublicAPI]
public static class SimdVector
{
    /// <summary>
    /// Creates a new <see cref="ISimdVector{TNumber}"/>.
    /// </summary>
    /// <typeparam name="TNumber">The number type of the <see cref="ISimdVector{TNumber}"/>.</typeparam>
    /// <returns>A new <see cref="ISimdVector{TNumber}"/>.</returns>
    [MethodImpl(ImplementationOptions.HotPath)]
    public static ISimdVector<TNumber> Create<TNumber>()
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (VectorGuard.IsSupported<TNumber>(out _))
        {
            return default(SimdScalarBackend<TNumber>);
        }

        return default(SimdScalarBackend<TNumber>);
    }

    /// <summary>
    /// Creates a vector filled with the given value.
    /// </summary>
    /// <param name="value">The value to fill the vector with.</param>
    /// <typeparam name="TNumber">The number type of the vector.</typeparam>
    /// <returns>A new vector filled with the given value.</returns>
    [MethodImpl(ImplementationOptions.HotPath)]
    public static ISimdVector<TNumber> Create<TNumber>(TNumber value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (VectorGuard.IsSupported<TNumber>(out var count))
        {
            Span<TNumber> span = stackalloc TNumber[count];
            span.Fill(value);

            return new SimdVectorBackend<TNumber>(span);
        }

        return new SimdScalarBackend<TNumber>(value);
    }

    /// <summary>
    /// Creates a new <see cref="ISimdVector{TNumber}"/> with the given value and vector strategy.
    /// </summary>
    /// <param name="value">The value to fill the vector with.</param>
    /// <param name="vectorStrategy">The vector strategy to use.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ISimdVector{TNumber}"/>.</typeparam>
    /// <returns>A new <see cref="ISimdVector{TNumber}"/>.</returns>
    [MethodImpl(ImplementationOptions.HotPath)]
    public static ISimdVector<TNumber> Create<TNumber>(TNumber value, VectorStrategy vectorStrategy)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        if (vectorStrategy == VectorStrategy.Vector && VectorGuard.IsSupported<TNumber>(out _))
        {
            Span<TNumber> span = stackalloc TNumber[Vector<TNumber>.Count];
            span.Fill(value);

            return new SimdVectorBackend<TNumber>(span);
        }

        return new SimdScalarBackend<TNumber>(value);
    }

    /// <summary>
    /// Loads a new <see cref="ISimdVector{TNumber}"/> from a source pointer as a scalar.
    /// </summary>
    /// <param name="value">The source pointer.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ISimdVector{TNumber}"/>.</typeparam>
    /// <returns>A new <see cref="ISimdVector{TNumber}"/>.</returns>
    [MethodImpl(ImplementationOptions.HotPath)]
    public static ISimdVector<TNumber> LoadScalar<TNumber>(TNumber value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return new SimdScalarBackend<TNumber>(value);
    }

    /// <summary>
    /// Loads a new <see cref="ISimdVector{TNumber}"/> from a source pointer.
    /// </summary>
    /// <param name="source">The source pointer.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ISimdVector{TNumber}"/>.</typeparam>
    /// <returns>A new <see cref="ISimdVector{TNumber}"/>.</returns>
    [MethodImpl(ImplementationOptions.HotPath)]
    public static unsafe ISimdVector<TNumber> UnsafeLoad<TNumber>(TNumber* source)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (VectorGuard.IsSupported<TNumber>(out _))
        {
            return new SimdVectorBackend<TNumber>(source);
        }

        return new SimdScalarBackend<TNumber>(source[0]);
    }

    /// <summary>
    /// Gets the number of elements in a SIMD vector.
    /// </summary>
    /// <typeparam name="TNumber">The number type of the <see cref="ISimdVector{TNumber}"/>.</typeparam>
    /// <returns>The number of elements in a SIMD vector.</returns>
    [MethodImpl(ImplementationOptions.HotPath)]
    public static int Count<TNumber>()
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (VectorGuard.IsSupported<TNumber>(out var count))
        {
            return count;
        }

        return 1;
    }

    /// <summary>
    /// Loads a new <see cref="ISimdVector{TNumber}"/> from a source memory block at the given index
    /// using the given vector strategy.
    /// </summary>
    /// <param name="memoryBlock">The input memory block.</param>
    /// <param name="index">The index to load from.</param>
    /// <param name="vectorStrategy">The vector strategy to use.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ISimdVector{TNumber}"/>.</typeparam>
    /// <returns>A new <see cref="ISimdVector{TNumber}"/>.</returns>
    [MethodImpl(ImplementationOptions.FastPath)]
    public static ISimdVector<TNumber> Load<TNumber>(SystemMemoryBlock<TNumber> memoryBlock, long index, VectorStrategy vectorStrategy)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        if (vectorStrategy is VectorStrategy.Scalar)
        {
            return LoadScalar(memoryBlock[index]);
        }

        if (VectorGuard.IsSupported<TNumber>(out _))
        {
            return memoryBlock.UnsafeGetVectorUnchecked<TNumber>(index);
        }

        return LoadScalar(memoryBlock[index]);
    }

    /// <summary>
    /// Loads a new <see cref="ISimdVector{TNumber}"/> from a source span.
    /// </summary>
    /// <param name="span">The source span.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ISimdVector{TNumber}"/>.</typeparam>
    /// <returns>A new <see cref="ISimdVector{TNumber}"/>.</returns>
    /// <exception cref="ArgumentException">Thrown if the span length is not 1 for scalar types.</exception>
    public static ISimdVector<TNumber> Load<TNumber>(Span<TNumber> span)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (VectorGuard.IsSupported<TNumber>(out _))
        {
            return new SimdVectorBackend<TNumber>(span);
        }

        if (span.Length != 1)
        {
            throw new ArgumentException("Span length must be 1 for scalar types.");
        }

        return new SimdScalarBackend<TNumber>(span[0]);
    }
}