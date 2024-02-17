// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Comparison;

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
        if (other is not SimdScalarBackend<TNumber> scalar)
        {
            throw new InvalidOperationException($"Cannot operate on a {typeof(SimdScalarBackend<TNumber>)} and a {other.GetType()}.");
        }

        return new SimdScalarBackend<TNumber>(_scalar + scalar._scalar);
    }

    public ISimdVector<TNumber> Subtract(ISimdVector<TNumber> other)
    {
        if (other is not SimdScalarBackend<TNumber> scalar)
        {
            throw new InvalidOperationException($"Cannot operate on a {typeof(SimdScalarBackend<TNumber>)} and a {other.GetType()}.");
        }

        return new SimdScalarBackend<TNumber>(_scalar - scalar._scalar);
    }

    public ISimdVector<TNumber> Multiply(ISimdVector<TNumber> other)
    {
        if (other is not SimdScalarBackend<TNumber> scalar)
        {
            throw new InvalidOperationException($"Cannot operate on a {typeof(SimdScalarBackend<TNumber>)} and a {other.GetType()}.");
        }

        return new SimdScalarBackend<TNumber>(_scalar * scalar._scalar);
    }

    public ISimdVector<TNumber> Divide(ISimdVector<TNumber> other)
    {
        if (other is not SimdScalarBackend<TNumber> scalar)
        {
            throw new InvalidOperationException($"Cannot operate on a {typeof(SimdScalarBackend<TNumber>)} and a {other.GetType()}.");
        }

        return new SimdScalarBackend<TNumber>(_scalar / scalar._scalar);
    }

    public ISimdVector<TNumber> Sqrt()
    {
        return _scalar switch
        {
            byte b => new SimdScalarBackend<TNumber>((TNumber)(object)(byte)MathF.Sqrt(b)),
            double d => new SimdScalarBackend<TNumber>((TNumber)(object)Math.Sqrt(d)),
            short s => new SimdScalarBackend<TNumber>((TNumber)(object)(short)MathF.Sqrt(s)),
            int i => new SimdScalarBackend<TNumber>((TNumber)(object)(int)Math.Sqrt(i)),
            long l => new SimdScalarBackend<TNumber>((TNumber)(object)(long)Math.Sqrt(l)),
            nint n => new SimdScalarBackend<TNumber>((TNumber)(object)(nint)Math.Sqrt(n)),
            nuint n => new SimdScalarBackend<TNumber>((TNumber)(object)(nuint)Math.Sqrt(n)),
            sbyte s => new SimdScalarBackend<TNumber>((TNumber)(object)(sbyte)MathF.Sqrt(s)),
            float f => new SimdScalarBackend<TNumber>((TNumber)(object)MathF.Sqrt(f)),
            ushort u => new SimdScalarBackend<TNumber>((TNumber)(object)(ushort)MathF.Sqrt(u)),
            uint u => new SimdScalarBackend<TNumber>((TNumber)(object)(uint)Math.Sqrt(u)),
            ulong u => (ISimdVector<TNumber>)new SimdScalarBackend<TNumber>((TNumber)(object)(ulong)Math.Sqrt(u)),
            _ => throw new NotSupportedException("Type not supported for square root.")
        };
    }

    public ISimdVector<TNumber> Abs()
    {
        return new SimdScalarBackend<TNumber>(TNumber.CreateChecked(Math.Abs(double.CreateChecked(_scalar))));
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
        if (other is not SimdScalarBackend<TNumber> scalar)
        {
            throw new InvalidOperationException($"Cannot operate on a {typeof(SimdScalarBackend<TNumber>)} and a {other.GetType()}.");
        }

        return new SimdScalarBackend<TNumber>(TNumber.Min(_scalar, scalar._scalar));
    }

    public ISimdVector<TNumber> Clamp(ISimdVector<TNumber> min, ISimdVector<TNumber> max)
    {
        if (min is not SimdScalarBackend<TNumber> minScalar)
        {
            throw new InvalidOperationException($"Cannot operate on a {typeof(SimdScalarBackend<TNumber>)} and a {min.GetType()}.");
        }

        if (max is not SimdScalarBackend<TNumber> maxScalar)
        {
            throw new InvalidOperationException($"Cannot operate on a {typeof(SimdScalarBackend<TNumber>)} and a {max.GetType()}.");
        }

        return new SimdScalarBackend<TNumber>(TNumber.Clamp(_scalar, minScalar._scalar, maxScalar._scalar));
    }

    public TNumber Dot(ISimdVector<TNumber> other)
    {
        if (other is not SimdScalarBackend<TNumber> scalar)
        {
            throw new InvalidOperationException($"Cannot operate on a {typeof(SimdScalarBackend<TNumber>)} and a {other.GetType()}.");
        }

        return _scalar * scalar._scalar;
    }

    public TNumber Sum()
    {
        return _scalar;
    }

    public ISimdVector<TNumber> SquareDifference(ISimdVector<TNumber> other)
    {
        if (other is not SimdScalarBackend<TNumber> otherVector)
        {
            throw new InvalidOperationException($"Cannot operate on a {typeof(SimdScalarBackend<TNumber>)} and a {other.GetType()}.");
        }

        var sum = _scalar + otherVector._scalar;

        return new SimdScalarBackend<TNumber>(sum * sum);
    }

    public ISimdVector<TNumber> CreateDuplicateZeroed()
    {
        return default(SimdScalarBackend<TNumber>);
    }

    public void CopyTo(Span<TNumber> span)
    {
        if (span.Length != 1)
        {
            throw new ArgumentException("Span length must be 1 for scalar types.");
        }

        span[0] = _scalar;
    }

    public ISimdVector<TNumber> CreateWith(Span<TNumber> values)
    {
        if (values.Length != Count)
        {
            throw new ArgumentOutOfRangeException(nameof(values), $"Values contains {values.Length} elements but expected {Count}.");
        }

        return new SimdScalarBackend<TNumber>(values[0]);
    }
}