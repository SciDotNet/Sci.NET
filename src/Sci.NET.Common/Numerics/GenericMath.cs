// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;

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
    /// Gets the maximum value for the specified number type.
    /// </summary>
    /// <typeparam name="TNumber">The number type to get the maximum value for.</typeparam>
    /// <returns>The maximum value for the specified number type.</returns>
    /// <exception cref="NotSupportedException">Thrown if the number type is not supported.</exception>
    public static TNumber MaxValue<TNumber>()
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TNumber.Zero switch
        {
            sbyte => TNumber.CreateChecked(sbyte.MaxValue),
            byte => TNumber.CreateChecked(byte.MaxValue),
            short => TNumber.CreateChecked(short.MaxValue),
            ushort => TNumber.CreateChecked(ushort.MaxValue),
            int => TNumber.CreateChecked(int.MaxValue),
            uint => TNumber.CreateChecked(uint.MaxValue),
            long => TNumber.CreateChecked(long.MaxValue),
            ulong => TNumber.CreateChecked(ulong.MaxValue),
            float => TNumber.CreateChecked(float.MaxValue),
            double => TNumber.CreateChecked(double.MaxValue),
            BFloat16 => TNumber.CreateChecked(BFloat16.MaxValue),
            Half => TNumber.CreateChecked(Half.MaxValue),
            _ => throw new NotSupportedException("Type not supported for MaxValue."),
        };
    }

    /// <summary>
    /// Gets the minimum value for the specified number type.
    /// </summary>
    /// <typeparam name="TNumber">The number type to get the minimum value for.</typeparam>
    /// <returns>The minimum value for the specified number type.</returns>
    /// <exception cref="NotSupportedException">Thrown if the number type is not supported.</exception>
    public static TNumber MinValue<TNumber>()
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TNumber.Zero switch
        {
            sbyte => TNumber.CreateChecked(sbyte.MinValue),
            byte => TNumber.CreateChecked(byte.MinValue),
            short => TNumber.CreateChecked(short.MinValue),
            ushort => TNumber.CreateChecked(ushort.MinValue),
            int => TNumber.CreateChecked(int.MinValue),
            uint => TNumber.CreateChecked(uint.MinValue),
            long => TNumber.CreateChecked(long.MinValue),
            ulong => TNumber.CreateChecked(ulong.MinValue),
            float => TNumber.CreateChecked(float.MinValue),
            double => TNumber.CreateChecked(double.MinValue),
            BFloat16 => TNumber.CreateChecked(BFloat16.MinValue),
            Half => TNumber.CreateChecked(Half.MinValue),
            _ => throw new NotSupportedException("Type not supported for MaxValue."),
        };
    }
}