// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Sci.NET.Common;

/// <summary>
/// An unsigned integer with the size of a pointer on the current platform.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
[DebuggerDisplay("{_value}")]
[PublicAPI]
public readonly struct SizeT : IEquatable<SizeT>, IEquatable<int>
{
    private readonly nuint _value;

    /// <summary>
    /// Initializes a new instance of the <see cref="SizeT"/> struct.
    /// </summary>
    /// <param name="value">The value of the <see cref="SizeT"/>.</param>
    public SizeT(nint value)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(value, 0);

        _value = (nuint)value;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SizeT"/> struct.
    /// </summary>
    /// <param name="value">The value of the <see cref="SizeT"/>.</param>
    public SizeT(nuint value)
    {
        _value = value;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SizeT"/> struct.
    /// </summary>
    /// <param name="value">The value of the <see cref="SizeT"/>.</param>
    public SizeT(int value)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(value, 0);
        _value = (nuint)value;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SizeT"/> struct.
    /// </summary>
    /// <param name="value">The value of the <see cref="SizeT"/>.</param>
    public SizeT(uint value)
    {
        _value = value;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SizeT"/> struct.
    /// </summary>
    /// <param name="value">The value of the <see cref="SizeT"/>.</param>
    public unsafe SizeT(long value)
    {
        _value = (nuint)(void*)value;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SizeT"/> struct.
    /// </summary>
    /// <param name="value">The value of the <see cref="SizeT"/>.</param>
    public SizeT(ulong value)
    {
        _value = (nuint)value;
    }

    /// <summary>
    /// Gets a <see cref="SizeT"/> representing zero.
    /// </summary>
    public static SizeT Zero => new(0);

    /// <inheritdoc cref="FromInt64"/>
    public static implicit operator SizeT(long value)
    {
        return new(value);
    }

    /// <summary>
    /// Determines if the <paramref name="left"/> operand is equal
    /// to the <paramref name="right"/> operand.
    /// </summary>
    /// <param name="left">Left operand.</param>
    /// <param name="right">Right operand.</param>
    /// <returns><c>true</c> if the two values are equal, else <c>false</c>.</returns>
    public static bool operator ==(SizeT left, SizeT right)
    {
        return left.Equals(right);
    }

    /// <summary>
    /// Determines if the <paramref name="left"/> operand is not equal
    /// to the <paramref name="right"/> operand.
    /// </summary>
    /// <param name="left">Left operand.</param>
    /// <param name="right">Right operand.</param>
    /// <returns><c>true</c> if the two values are not equal, else <c>false</c>.</returns>
    public static bool operator !=(SizeT left, SizeT right)
    {
        return !(left == right);
    }

    /// <summary>
    /// Determines if the <paramref name="left"/> operand is less than the <paramref name="right"/> operand.
    /// </summary>
    /// <param name="left">Left operand.</param>
    /// <param name="right">Right operand.</param>
    /// <returns><c>true</c> if the <paramref name="left"/> operand is less than the <paramref name="right"/> operand, else <c>false</c>.</returns>
    public static bool operator <(SizeT left, SizeT right)
    {
        return left._value < right._value;
    }

    /// <summary>
    /// Determines if the <paramref name="left"/> operand is greater than the <paramref name="right"/> operand.
    /// </summary>
    /// <param name="left">Left operand.</param>
    /// <param name="right">Right operand.</param>
    /// <returns><c>true</c> if the <paramref name="left"/> operand is greater than the <paramref name="right"/> operand, else <c>false</c>.</returns>
    public static bool operator >(SizeT left, SizeT right)
    {
        return left._value > right._value;
    }

    /// <summary>
    /// Converts a <see cref="long"/> to a <see cref="SizeT"/>.
    /// </summary>
    /// <param name="value">The value of to convert to <see cref="SizeT"/>.</param>
    /// <returns>A <see cref="SizeT"/> with the value given by <paramref name="value"/>.</returns>
    public static SizeT FromInt64(long value)
    {
        return new(value);
    }

    /// <summary>
    /// Converts the value of the current <see cref="SizeT"/> to a <see cref="UIntPtr"/>.
    /// </summary>
    /// <returns>The value of the <see cref="SizeT"/> as a <see cref="UIntPtr"/>.</returns>
    public nuint ToUIntPtr()
    {
        return _value;
    }

    /// <summary>
    /// Converts the value of the current <see cref="SizeT"/> to a <see cref="IntPtr"/>.
    /// </summary>
    /// <returns>The value of the <see cref="SizeT"/> as an <see cref="IntPtr"/>.</returns>
    public nint ToIntPtr()
    {
        return (nint)_value;
    }

    /// <inheritdoc />
    public bool Equals(SizeT other)
    {
        return _value.Equals(other._value);
    }

    /// <inheritdoc />
    public bool Equals(int other)
    {
        return _value == (nuint)other;
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        return obj is SizeT other && Equals(other);
    }

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        return _value.GetHashCode();
    }

    /// <summary>
    /// Converts the value of the current <see cref="SizeT"/> to a <see cref="long"/>.
    /// </summary>
    /// <returns>The current instance as a <see cref="long"/>.</returns>
    public long ToInt64()
    {
        return (long)_value;
    }
}