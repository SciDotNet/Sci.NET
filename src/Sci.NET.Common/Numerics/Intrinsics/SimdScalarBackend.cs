// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Comparison;
using Sci.NET.Common.Numerics.Intrinsics.Exceptions;

namespace Sci.NET.Common.Numerics.Intrinsics;

internal readonly struct SimdScalarBackend<TNumber> : ISimdVector<TNumber>, IValueEquatable<SimdScalarBackend<TNumber>>
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly TNumber _scalar;

    public SimdScalarBackend(TNumber source)
    {
        _scalar = source;
    }

    public int Count => 1;

    public TNumber this[int index] => index == 0 ? _scalar : throw new ArgumentOutOfRangeException(nameof(index), "Scalar types only have one element.");

    public static bool operator ==(SimdScalarBackend<TNumber> left, SimdScalarBackend<TNumber> right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(SimdScalarBackend<TNumber> left, SimdScalarBackend<TNumber> right)
    {
        return !(left == right);
    }

    public bool Equals(SimdScalarBackend<TNumber> other)
    {
        return _scalar.Equals(other._scalar);
    }

    public override bool Equals(object? obj)
    {
        return obj is SimdScalarBackend<TNumber> other && Equals(other);
    }

    public override int GetHashCode()
    {
        return _scalar.GetHashCode();
    }

    public ISimdVector<TNumber> Add(ISimdVector<TNumber> other)
    {
        InvalidVectorOperationException.ThrowIfNotOfType<SimdScalarBackend<TNumber>, TNumber>(other);

        return new SimdScalarBackend<TNumber>(_scalar + ((SimdScalarBackend<TNumber>)other)._scalar);
    }

    public ISimdVector<TNumber> Subtract(ISimdVector<TNumber> other)
    {
        InvalidVectorOperationException.ThrowIfNotOfType<SimdScalarBackend<TNumber>, TNumber>(other);

        return new SimdScalarBackend<TNumber>(_scalar - ((SimdScalarBackend<TNumber>)other)._scalar);
    }

    public ISimdVector<TNumber> Multiply(ISimdVector<TNumber> other)
    {
        InvalidVectorOperationException.ThrowIfNotOfType<SimdScalarBackend<TNumber>, TNumber>(other);

        return new SimdScalarBackend<TNumber>(_scalar * ((SimdScalarBackend<TNumber>)other)._scalar);
    }

    public ISimdVector<TNumber> Divide(ISimdVector<TNumber> other)
    {
        InvalidVectorOperationException.ThrowIfNotOfType<SimdScalarBackend<TNumber>, TNumber>(other);

        return new SimdScalarBackend<TNumber>(_scalar / ((SimdScalarBackend<TNumber>)other)._scalar);
    }

    public ISimdVector<TNumber> Sqrt()
    {
        return new SimdScalarBackend<TNumber>(GenericMath.Sqrt(_scalar));
    }

    public ISimdVector<TNumber> Abs()
    {
        return GenericMath.Abs(_scalar);
    }

    public ISimdVector<TNumber> Negate()
    {
        return new SimdScalarBackend<TNumber>(-_scalar);
    }

    public ISimdVector<TNumber> Max(ISimdVector<TNumber> other)
    {
        return new SimdScalarBackend<TNumber>(TNumber.Max(_scalar, ((SimdScalarBackend<TNumber>)other)._scalar));
    }

    public ISimdVector<TNumber> Min(ISimdVector<TNumber> other)
    {
        InvalidVectorOperationException.ThrowIfNotOfType<SimdScalarBackend<TNumber>, TNumber>(other);

        return new SimdScalarBackend<TNumber>(TNumber.Min(_scalar, ((SimdScalarBackend<TNumber>)other)._scalar));
    }

    public ISimdVector<TNumber> Clamp(ISimdVector<TNumber> min, ISimdVector<TNumber> max)
    {
        InvalidVectorOperationException.ThrowIfNotOfType<SimdScalarBackend<TNumber>, TNumber>(min, max);

        return new SimdScalarBackend<TNumber>(TNumber.Clamp(_scalar, ((SimdScalarBackend<TNumber>)min)._scalar, ((SimdScalarBackend<TNumber>)max)._scalar));
    }

    public TNumber Dot(ISimdVector<TNumber> other)
    {
        InvalidVectorOperationException.ThrowIfNotOfType<SimdScalarBackend<TNumber>, TNumber>(other);

        return _scalar * ((SimdScalarBackend<TNumber>)other)._scalar;
    }

    public TNumber Sum()
    {
        return _scalar;
    }

    public ISimdVector<TNumber> SquareDifference(ISimdVector<TNumber> other)
    {
        InvalidVectorOperationException.ThrowIfNotOfType<SimdScalarBackend<TNumber>, TNumber>(other);

        var sum = _scalar + ((SimdScalarBackend<TNumber>)other)._scalar;

        return new SimdScalarBackend<TNumber>(sum * sum);
    }

    public ISimdVector<TNumber> CreateDuplicateZeroed()
    {
        return default(SimdScalarBackend<TNumber>);
    }

    public void CopyTo(Span<TNumber> span)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(span.Length, 1);

        span[0] = _scalar;
    }

    public ISimdVector<TNumber> CreateWith(Span<TNumber> values)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(values.Length, Count);

        return new SimdScalarBackend<TNumber>(values[0]);
    }

    public TNumber MaxElement()
    {
        return _scalar;
    }

    public TNumber MinElement()
    {
        return _scalar;
    }
}