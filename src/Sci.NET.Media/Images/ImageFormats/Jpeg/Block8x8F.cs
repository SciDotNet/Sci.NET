// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sci.NET.Common.Comparison;
using Sci.NET.Common.Performance;

namespace Sci.NET.Media.Images.ImageFormats.Jpeg;

/// <summary>
/// A block of 8x8 floating point values.
/// </summary>
[PublicAPI]
[SuppressMessage("Design", "CA1045:Do not pass types by reference", Justification = "Performance")]
[StructLayout(LayoutKind.Explicit)]
public readonly struct Block8X8F : IValueEquatable<Block8X8F>
{
    /// <summary>
    /// The number of elements in a block.
    /// </summary>
    private const int Size = 8 * 8;

    [FieldOffset(sizeof(float) * 8 * 0)] private readonly Vector256<float> _r0;
    [FieldOffset(sizeof(float) * 8 * 1)] private readonly Vector256<float> _r1;
    [FieldOffset(sizeof(float) * 8 * 2)] private readonly Vector256<float> _r2;
    [FieldOffset(sizeof(float) * 8 * 3)] private readonly Vector256<float> _r3;
    [FieldOffset(sizeof(float) * 8 * 4)] private readonly Vector256<float> _r4;
    [FieldOffset(sizeof(float) * 8 * 5)] private readonly Vector256<float> _r5;
    [FieldOffset(sizeof(float) * 8 * 6)] private readonly Vector256<float> _r6;
    [FieldOffset(sizeof(float) * 8 * 7)] private readonly Vector256<float> _r7;

    /// <summary>
    /// Initializes a new instance of the <see cref="Block8X8F"/> struct.
    /// </summary>
    public Block8X8F()
    {
        _r0 = default;
        _r1 = default;
        _r2 = default;
        _r3 = default;
        _r4 = default;
        _r5 = default;
        _r6 = default;
        _r7 = default;
    }

    /// <summary>
    /// Gets the element at the specified linear index.
    /// </summary>
    /// <param name="linearIndex">The index to query.</param>
    public float this[int linearIndex]
    {
        get => GetElement(linearIndex);
        set => SetElement(linearIndex, value);
    }

    /// <summary>
    /// Gets the element at the specified 2D indices.
    /// </summary>
    /// <param name="row">The row index.</param>
    /// <param name="column">The column index.</param>
    public float this[int row, int column]
    {
        get => this[(row * 8) + column];
        set => this[(row * 8) + column] = value;
    }

    /// <inheritdoc />
    [MethodImpl(ImplementationOptions.FastPath)]
    public static bool operator ==(Block8X8F left, Block8X8F right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    [MethodImpl(ImplementationOptions.FastPath)]
    public static bool operator !=(Block8X8F left, Block8X8F right)
    {
        return !left.Equals(right);
    }

    /// <summary>
    /// Multiplies the two blocks together.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The product of the two operands.</returns>
    public static Block8X8F operator *(Block8X8F left, Block8X8F right)
    {
        return Multiply(left, right);
    }

    /// <summary>
    /// Adds the two blocks together.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The sum of the two operands.</returns>
    public static Block8X8F operator +(Block8X8F left, Block8X8F right)
    {
        return Add(left, right);
    }

    /// <summary>
    /// Adds the block to the other <see cref="Block8X8F"/> element-wise.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The sum of the left and right operands.</returns>
    [MethodImpl(ImplementationOptions.HotPath)]
    public static Block8X8F Add(Block8X8F left, Block8X8F right)
    {
        var result = default(Block8X8F);

        if (Avx.IsSupported)
        {
            Unsafe.AsRef(result._r0) = Avx.Add(left._r0, right._r0);
            Unsafe.AsRef(result._r1) = Avx.Add(left._r1, right._r1);
            Unsafe.AsRef(result._r2) = Avx.Add(left._r2, right._r2);
            Unsafe.AsRef(result._r3) = Avx.Add(left._r3, right._r3);
            Unsafe.AsRef(result._r4) = Avx.Add(left._r4, right._r4);
            Unsafe.AsRef(result._r5) = Avx.Add(left._r5, right._r5);
            Unsafe.AsRef(result._r6) = Avx.Add(left._r6, right._r6);
            Unsafe.AsRef(result._r7) = Avx.Add(left._r7, right._r7);
        }
        else
        {
            Unsafe.AsRef(result._r0) += left._r0 * right._r0;
            Unsafe.AsRef(result._r1) += left._r1 * right._r1;
            Unsafe.AsRef(result._r2) += left._r2 * right._r2;
            Unsafe.AsRef(result._r3) += left._r3 * right._r3;
            Unsafe.AsRef(result._r4) += left._r4 * right._r4;
            Unsafe.AsRef(result._r5) += left._r5 * right._r5;
            Unsafe.AsRef(result._r6) += left._r6 * right._r6;
            Unsafe.AsRef(result._r7) += left._r7 * right._r7;
        }

        return result;
    }

    /// <summary>
    /// Multiplies the block by the given value.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The product of the left and right operands.</returns>
    [MethodImpl(ImplementationOptions.HotPath)]
    public static Block8X8F Multiply(ref Block8X8F left, float right)
    {
        var scalar = Vector256.Create(right);
        var result = default(Block8X8F);

        if (Avx.IsSupported)
        {
            Unsafe.AsRef(result._r0) = Avx.Multiply(left._r0, scalar);
            Unsafe.AsRef(result._r1) = Avx.Multiply(left._r1, scalar);
            Unsafe.AsRef(result._r2) = Avx.Multiply(left._r2, scalar);
            Unsafe.AsRef(result._r3) = Avx.Multiply(left._r3, scalar);
            Unsafe.AsRef(result._r4) = Avx.Multiply(left._r4, scalar);
            Unsafe.AsRef(result._r5) = Avx.Multiply(left._r5, scalar);
            Unsafe.AsRef(result._r6) = Avx.Multiply(left._r6, scalar);
            Unsafe.AsRef(result._r7) = Avx.Multiply(left._r7, scalar);
        }
        else
        {
            Unsafe.AsRef(result._r1) *= scalar;
            Unsafe.AsRef(result._r2) *= scalar;
            Unsafe.AsRef(result._r3) *= scalar;
            Unsafe.AsRef(result._r4) *= scalar;
            Unsafe.AsRef(result._r5) *= scalar;
            Unsafe.AsRef(result._r6) *= scalar;
            Unsafe.AsRef(result._r7) *= scalar;
        }

        return result;
    }

    /// <summary>
    /// Multiplies the block by the other <see cref="Block8X8F"/> element-wise.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The product of the left and right operands.</returns>
    [MethodImpl(ImplementationOptions.HotPath)]
    public static Block8X8F Multiply(Block8X8F left, Block8X8F right)
    {
        var result = default(Block8X8F);

        if (Avx.IsSupported)
        {
            Unsafe.AsRef(result._r0) = Avx.Multiply(left._r0, right._r0);
            Unsafe.AsRef(result._r1) = Avx.Multiply(left._r1, right._r1);
            Unsafe.AsRef(result._r2) = Avx.Multiply(left._r2, right._r2);
            Unsafe.AsRef(result._r3) = Avx.Multiply(left._r3, right._r3);
            Unsafe.AsRef(result._r4) = Avx.Multiply(left._r4, right._r4);
            Unsafe.AsRef(result._r5) = Avx.Multiply(left._r5, right._r5);
            Unsafe.AsRef(result._r6) = Avx.Multiply(left._r6, right._r6);
            Unsafe.AsRef(result._r7) = Avx.Multiply(left._r7, right._r7);
        }
        else
        {
            Unsafe.AsRef(result._r0) *= left._r0 * right._r0;
            Unsafe.AsRef(result._r1) *= left._r1 * right._r1;
            Unsafe.AsRef(result._r2) *= left._r2 * right._r2;
            Unsafe.AsRef(result._r3) *= left._r3 * right._r3;
            Unsafe.AsRef(result._r4) *= left._r4 * right._r4;
            Unsafe.AsRef(result._r5) *= left._r5 * right._r5;
            Unsafe.AsRef(result._r6) *= left._r6 * right._r6;
            Unsafe.AsRef(result._r7) *= left._r7 * right._r7;
        }

        return result;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    [MethodImpl(ImplementationOptions.FastPath)]
    public override bool Equals(object? obj)
    {
        return obj is Block8X8F other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    [MethodImpl(ImplementationOptions.FastPath)]
    public bool Equals(Block8X8F other)
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

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    [MethodImpl(ImplementationOptions.FastPath)]
    public override int GetHashCode()
    {
        var hashCode = default(HashCode);
        hashCode.Add(_r0);
        hashCode.Add(_r1);
        hashCode.Add(_r2);
        hashCode.Add(_r3);
        hashCode.Add(_r4);
        hashCode.Add(_r5);
        hashCode.Add(_r6);
        hashCode.Add(_r7);
        return hashCode.ToHashCode();
    }

    [MethodImpl(ImplementationOptions.FastPath)]
    private void SetElement(int linearIndex, float value)
    {
        if (linearIndex is < 0 or >= Size)
        {
            throw new ArgumentOutOfRangeException(nameof(linearIndex), "The linear index must be in the range 0-63.");
        }

        ref var instance = ref Unsafe.As<Block8X8F, float>(ref Unsafe.AsRef(this));
        Unsafe.Add(ref instance, linearIndex) = value;
    }

    [MethodImpl(ImplementationOptions.FastPath)]
    private float GetElement(int linearIndex)
    {
        if (linearIndex is < 0 or >= Size)
        {
            throw new ArgumentOutOfRangeException(nameof(linearIndex), "The linear index must be in the range 0-63.");
        }

        ref var instance = ref Unsafe.As<Block8X8F, float>(ref Unsafe.AsRef(this));
        return Unsafe.Add(ref instance, linearIndex);
    }
}