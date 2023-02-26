// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Sci.NET.Common.Comparison;
using Sci.NET.Common.Performance;

namespace Sci.NET.Media.Images.ImageFormats.Jpeg;

/// <summary>
/// A block of 8x8 int-16 values.
/// </summary>
[PublicAPI]
[SuppressMessage("Design", "CA1045:Do not pass types by reference", Justification = "Performance")]
[StructLayout(LayoutKind.Explicit)]
public struct Block8X8 : IValueEquatable<Block8X8>
{
    /// <summary>
    /// The number of elements in a block.
    /// </summary>
    private const int Size = 8 * 8;

    [FieldOffset(sizeof(short) * 8 * 0)] private readonly Vector128<short> _r0;
    [FieldOffset(sizeof(short) * 8 * 1)] private readonly Vector128<short> _r1;
    [FieldOffset(sizeof(short) * 8 * 2)] private readonly Vector128<short> _r2;
    [FieldOffset(sizeof(short) * 8 * 3)] private readonly Vector128<short> _r3;
    [FieldOffset(sizeof(short) * 8 * 4)] private readonly Vector128<short> _r4;
    [FieldOffset(sizeof(short) * 8 * 5)] private readonly Vector128<short> _r5;
    [FieldOffset(sizeof(short) * 8 * 6)] private readonly Vector128<short> _r6;
    [FieldOffset(sizeof(short) * 8 * 7)] private readonly Vector128<short> _r7;

    /// <summary>
    /// Gets the element at the specified linear index.
    /// </summary>
    /// <param name="linearIndex">The index to query.</param>
    public short this[int linearIndex]
    {
        get => GetElement(linearIndex);
        set => SetElement(linearIndex, value);
    }

    /// <summary>
    /// Gets the element at the specified 2D indices.
    /// </summary>
    /// <param name="row">The row index.</param>
    /// <param name="column">The column index.</param>
    public short this[int row, int column]
    {
        get => this[(row * 8) + column];
        set => this[(row * 8) + column] = value;
    }

    /// <inheritdoc />
    public static bool operator ==(Block8X8 left, Block8X8 right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(Block8X8 left, Block8X8 right)
    {
        return !left.Equals(right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(Block8X8 other)
    {
        return _r0.Equals(other._r0) &&
               _r1.Equals(other._r1) &&
               _r2.Equals(other._r2) &&
               _r3.Equals(other._r3) &&
               _r4.Equals(other._r4) &&
               _r5.Equals(other._r5) &&
               _r6.Equals(other._r6) &&
               _r7.Equals(other._r7);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is Block8X8 other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
    }

    [MethodImpl(ImplementationOptions.FastPath)]
    private short GetElement(int linearIndex)
    {
        if (linearIndex is < 0 or >= Size)
        {
            throw new ArgumentOutOfRangeException(nameof(linearIndex), "The linear index must be in the range 0-63.");
        }

        ref var instance = ref Unsafe.As<Block8X8, short>(ref Unsafe.AsRef(this));
        return Unsafe.Add(ref instance, linearIndex);
    }

    [MethodImpl(ImplementationOptions.FastPath)]
    private void SetElement(int linearIndex, short value)
    {
        if (linearIndex is < 0 or >= Size)
        {
            throw new ArgumentOutOfRangeException(nameof(linearIndex), "The linear index must be in the range 0-63.");
        }

        ref var instance = ref Unsafe.As<Block8X8, short>(ref Unsafe.AsRef(this));
        Unsafe.Add(ref instance, linearIndex) = value;
    }
}