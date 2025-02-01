// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;
using Sci.NET.Common.Numerics.Intrinsics;

namespace Sci.NET.Common.Numerics;

/// <summary>
/// A helper class for generic math operations.
/// </summary>
[PublicAPI]
public static class GenericMath
{
    /// <summary>
    /// Determines if the number type is floating point.
    /// </summary>
    /// <typeparam name="TNumber">The number type to test.</typeparam>
    /// <returns><c>true</c> if the number is a floating point type, else, <c>false</c>.</returns>
    public static bool IsFloatingPoint<TNumber>()
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TNumber.CreateChecked(0.01f) != TNumber.Zero;
    }

    /// <summary>
    /// Determines if the number type is signed.
    /// </summary>
    /// <typeparam name="TNumber">The number type to test.</typeparam>
    /// <returns><c>true</c> if the number is a signed type, else, <c>false</c>.</returns>
    public static bool IsSigned<TNumber>()
        where TNumber : unmanaged, INumber<TNumber>
    {
        return unchecked(TNumber.Zero - TNumber.One) < TNumber.Zero;
    }

    /// <summary>
    /// Gets the machine epsilon for the specified number type.
    /// </summary>
    /// <typeparam name="TNumber">The number type to get the epsilon for.</typeparam>
    /// <returns>The machine epsilon for the specified number type.</returns>
    public static unsafe TNumber Epsilon<TNumber>()
        where TNumber : unmanaged, INumber<TNumber>
    {
        var numBytes = Unsafe.SizeOf<TNumber>();
        var bytes = stackalloc byte[Unsafe.SizeOf<TNumber>()];

        if (BitConverter.IsLittleEndian)
        {
            bytes[0] = 0x01;
        }
        else
        {
            bytes[numBytes - 1] = 0x01;
        }

        return Unsafe.Read<TNumber>(bytes);
    }

    /// <summary>
    /// Finds the square root of the provided number.
    /// </summary>
    /// <param name="number">The number to find the square root of.</param>
    /// <typeparam name="TNumber">The number type.</typeparam>
    /// <returns>The square root of the provided number.</returns>
    /// <exception cref="NotSupportedException">Thrown if the number type is not supported.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static TNumber Sqrt<TNumber>(TNumber number)
    {
        return number switch
        {
            float f => (TNumber)(object)MathF.Sqrt(f),
            double d => (TNumber)(object)Math.Sqrt(d),
            BFloat16 b => (TNumber)(object)BFloat16.Sqrt(b),
            Half h => (TNumber)(object)Half.Sqrt(h),
            Complex c => (TNumber)(object)Complex.Sqrt(c),
            byte b => (TNumber)(object)(byte)MathF.Sqrt(b),
            short s => (TNumber)(object)(short)MathF.Sqrt(s),
            int i => (TNumber)(object)(int)Math.Sqrt(i),
            long l => (TNumber)(object)(long)Math.Sqrt(l),
            nint n => (TNumber)(object)(nint)Math.Sqrt(n),
            nuint n => (TNumber)(object)(nuint)Math.Sqrt(n),
            sbyte s => (TNumber)(object)(sbyte)MathF.Sqrt(s),
            ushort u => (TNumber)(object)(ushort)MathF.Sqrt(u),
            uint u => (TNumber)(object)(uint)Math.Sqrt(u),
            ulong u => (TNumber)(object)(ulong)Math.Sqrt(u),
            _ => throw new NotSupportedException("Type not supported for square root."),
        };
    }

    /// <summary>
    /// Finds a scaled epsilon for the specified number type. If the number type is a floating point type, the epsilon is multiplied by the specified multiplier.
    /// </summary>
    /// <typeparam name="TNumber">The number type to find the epsilon for.</typeparam>
    /// <param name="multiplier">The multiplier to use for the epsilon.</param>
    /// <returns>A small epsilon for the specified number type.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static TNumber ScaledEpsilon<TNumber>(int multiplier = 1000)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var epsilon = Epsilon<TNumber>();

        if (IsFloatingPoint<TNumber>())
        {
            epsilon *= TNumber.CreateChecked(multiplier);
        }

        return epsilon;
    }

    /// <summary>
    /// Finds the absolute value of the provided scalar.
    /// </summary>
    /// <param name="scalar">The scalar to find the absolute value of.</param>
    /// <typeparam name="TNumber">The number type.</typeparam>
    /// <returns>The absolute value of the provided scalar.</returns>
    /// <exception cref="NotSupportedException">Thrown if the number type is not supported.</exception>
    public static ISimdVector<TNumber> Abs<TNumber>(TNumber scalar)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return scalar switch
        {
            BFloat16 bfloat16 => new SimdScalarBackend<TNumber>((TNumber)(object)BFloat16.Abs(bfloat16)),
            Half half => new SimdScalarBackend<TNumber>((TNumber)(object)Half.Abs(half)),
            Complex complex => new SimdScalarBackend<TNumber>((TNumber)(object)Complex.Abs(complex)),
            float single => new SimdScalarBackend<TNumber>((TNumber)(object)float.Abs(single)),
            double @double => new SimdScalarBackend<TNumber>((TNumber)(object)double.Abs(@double)),
            byte byteValue => new SimdScalarBackend<TNumber>((TNumber)(object)byteValue),
            ushort ushortValue => new SimdScalarBackend<TNumber>((TNumber)(object)ushortValue),
            uint uintValue => new SimdScalarBackend<TNumber>((TNumber)(object)uintValue),
            ulong ulongValue => new SimdScalarBackend<TNumber>((TNumber)(object)ulongValue),
            sbyte sbyteValue => new SimdScalarBackend<TNumber>((TNumber)(object)sbyte.Abs(sbyteValue)),
            short shortValue => new SimdScalarBackend<TNumber>((TNumber)(object)short.Abs(shortValue)),
            int intValue => new SimdScalarBackend<TNumber>((TNumber)(object)int.Abs(intValue)),
            long longValue => new SimdScalarBackend<TNumber>((TNumber)(object)long.Abs(longValue)),
            nint nintValue => new SimdScalarBackend<TNumber>((TNumber)(object)nint.Abs(nintValue)),
            nuint nuintValue => new SimdScalarBackend<TNumber>((TNumber)(object)nuintValue),
            _ => throw new NotSupportedException("Type not supported for absolute value."),
        };
    }
}