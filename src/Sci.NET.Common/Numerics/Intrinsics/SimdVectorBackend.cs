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

    public int Count => Vector<TNumber>.Count;

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
        return new SimdVectorBackend<TNumber>(Vector.Add(_vector, ((SimdVectorBackend<TNumber>)other)._vector));
    }

    public ISimdVector<TNumber> Subtract(ISimdVector<TNumber> other)
    {
        return new SimdVectorBackend<TNumber>(Vector.Subtract(_vector, ((SimdVectorBackend<TNumber>)other)._vector));
    }

    public ISimdVector<TNumber> Multiply(ISimdVector<TNumber> other)
    {
        return new SimdVectorBackend<TNumber>(Vector.Multiply(_vector, ((SimdVectorBackend<TNumber>)other)._vector));
    }

    public ISimdVector<TNumber> Divide(ISimdVector<TNumber> other)
    {
        return new SimdVectorBackend<TNumber>(Vector.Divide(_vector, ((SimdVectorBackend<TNumber>)other)._vector));
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
        return new SimdVectorBackend<TNumber>(Vector.Negate(_vector));
    }

    public ISimdVector<TNumber> Max(ISimdVector<TNumber> other)
    {
        return new SimdVectorBackend<TNumber>(Vector.Max(_vector, ((SimdVectorBackend<TNumber>)other)._vector));
    }

    public ISimdVector<TNumber> Min(ISimdVector<TNumber> other)
    {
        return new SimdVectorBackend<TNumber>(Vector.Min(_vector, ((SimdVectorBackend<TNumber>)other)._vector));
    }

    public ISimdVector<TNumber> Clamp(ISimdVector<TNumber> min, ISimdVector<TNumber> max)
    {
        return new SimdVectorBackend<TNumber>(Vector.Min(Vector.Max(_vector, ((SimdVectorBackend<TNumber>)min)._vector), ((SimdVectorBackend<TNumber>)max)._vector));
    }

    public TNumber Dot(ISimdVector<TNumber> other)
    {
        return Vector.Dot(_vector, ((SimdVectorBackend<TNumber>)other)._vector);
    }

    public TNumber Sum()
    {
        return Vector.Sum(_vector);
    }

    public ISimdVector<TNumber> SquareDifference(ISimdVector<TNumber> other)
    {
        var difference = Vector.Subtract(_vector, ((SimdVectorBackend<TNumber>)other)._vector);

        return new SimdVectorBackend<TNumber>(Vector.Multiply(difference, difference));
    }

    public ISimdVector<TNumber> CreateDuplicateZeroed()
    {
        return default(SimdVectorBackend<TNumber>);
    }

    public void CopyTo(Span<TNumber> span)
    {
        if (span.Length != Vector<TNumber>.Count)
        {
            throw new ArgumentException("Span length must be equal to the vector count.");
        }

        _vector.CopyTo(span);
    }

    public ISimdVector<TNumber> CreateWith(Span<TNumber> values)
    {
        if (values.Length != Count)
        {
            throw new ArgumentOutOfRangeException(nameof(values), $"Values contains {values.Length} elements but expected {Count}.");
        }

        return new SimdVectorBackend<TNumber>(values);
    }

    public TNumber MaxElement()
    {
        var max = _vector[0];

        for (var i = 1; i < Count; i++)
        {
            max = TNumber.Max(max, _vector[i]);
        }

        return max;
    }

    public TNumber MinElement()
    {
        var min = _vector[0];

        for (var i = 1; i < Count; i++)
        {
            min = TNumber.Min(min, _vector[i]);
        }

        return min;
    }
}