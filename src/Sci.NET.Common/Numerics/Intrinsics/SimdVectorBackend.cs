// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Common.Numerics.Intrinsics;

internal readonly struct SimdVectorBackend<TNumber> : ISimdVector<TNumber>, IValueEquatable<SimdVectorBackend<TNumber>>
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly Vector<TNumber> _vector;

    public unsafe SimdVectorBackend(TNumber* source)
    {
        var span = new Span<TNumber>(source, Vector<TNumber>.Count);
        _vector = new Vector<TNumber>(span);
    }

    public SimdVectorBackend(Span<TNumber> source)
    {
        _vector = new Vector<TNumber>(source);
    }

    public SimdVectorBackend(Vector<TNumber> source)
    {
        _vector = source;
    }

    public TNumber this[int index] => _vector[index];

    public static bool operator ==(SimdVectorBackend<TNumber> left, SimdVectorBackend<TNumber> right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(SimdVectorBackend<TNumber> left, SimdVectorBackend<TNumber> right)
    {
        return !(left == right);
    }

    public bool Equals(SimdVectorBackend<TNumber> other)
    {
        return _vector.Equals(other._vector);
    }

    public override bool Equals(object? obj)
    {
        return obj is SimdVectorBackend<TNumber> other && Equals(other);
    }

    public override int GetHashCode()
    {
        return _vector.GetHashCode();
    }

    public ISimdVector<TNumber> Add(ISimdVector<TNumber> other)
    {
        if (other is not SimdVectorBackend<TNumber> vector)
        {
            throw new InvalidOperationException("Cannot add a scalar to a vector.");
        }

        return new SimdVectorBackend<TNumber>(_vector + vector._vector);
    }

    public ISimdVector<TNumber> Subtract(ISimdVector<TNumber> other)
    {
        if (other is not SimdVectorBackend<TNumber> vector)
        {
            throw new InvalidOperationException("Cannot subtract a scalar from a vector.");
        }

        return new SimdVectorBackend<TNumber>(_vector - vector._vector);
    }

    public ISimdVector<TNumber> Multiply(ISimdVector<TNumber> other)
    {
        if (other is not SimdVectorBackend<TNumber> vector)
        {
            throw new InvalidOperationException("Cannot multiply a vector by a scalar.");
        }

        return new SimdVectorBackend<TNumber>(_vector * vector._vector);
    }

    public ISimdVector<TNumber> Divide(ISimdVector<TNumber> other)
    {
        if (other is not SimdVectorBackend<TNumber> vector)
        {
            throw new InvalidOperationException("Cannot divide a vector by a scalar.");
        }

        return new SimdVectorBackend<TNumber>(_vector / vector._vector);
    }

    public ISimdVector<TNumber> Sqrt()
    {
        return new SimdVectorBackend<TNumber>(Vector.SquareRoot(_vector));
    }

    public ISimdVector<TNumber> Abs()
    {
        return new SimdVectorBackend<TNumber>(Vector.Abs(_vector));
    }

    public ISimdVector<TNumber> Negate()
    {
        return new SimdVectorBackend<TNumber>(-_vector);
    }

    public ISimdVector<TNumber> Max(ISimdVector<TNumber> other)
    {
        if (other is not SimdVectorBackend<TNumber> vector)
        {
            throw new InvalidOperationException("Cannot compare a vector to a scalar.");
        }

        return new SimdVectorBackend<TNumber>(Vector.Max(_vector, vector._vector));
    }

    public ISimdVector<TNumber> Min(ISimdVector<TNumber> other)
    {
        if (other is not SimdVectorBackend<TNumber> vector)
        {
            throw new InvalidOperationException("Cannot compare a vector to a scalar.");
        }

        return new SimdVectorBackend<TNumber>(Vector.Min(_vector, vector._vector));
    }

    public ISimdVector<TNumber> Clamp(ISimdVector<TNumber> min, ISimdVector<TNumber> max)
    {
        if (min is not SimdVectorBackend<TNumber> minVector)
        {
            throw new InvalidOperationException("Cannot compare a vector to a scalar.");
        }

        if (max is not SimdVectorBackend<TNumber> maxVector)
        {
            throw new InvalidOperationException("Cannot compare a vector to a scalar.");
        }

        return new SimdVectorBackend<TNumber>(Vector.Min(Vector.Max(_vector, minVector._vector), maxVector._vector));
    }

    public TNumber Dot(ISimdVector<TNumber> other)
    {
        if (other is not SimdVectorBackend<TNumber> vector)
        {
            throw new InvalidOperationException("Cannot compare a vector to a scalar.");
        }

        return Vector.Dot(_vector, vector._vector);
    }

    public TNumber Sum()
    {
        return Vector.Sum(_vector);
    }

    public void CopyTo(Span<TNumber> span)
    {
        if (span.Length != Vector<TNumber>.Count)
        {
            throw new ArgumentException("Span length must be equal to the vector count.");
        }

        _vector.CopyTo(span);
    }
}