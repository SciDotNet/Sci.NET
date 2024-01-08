// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Buffers.Binary;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Sci.NET.Common.Numerics;

/// <summary>
/// A 16-bit floating point number in the bfloat16 format derived from the Open Neural Network Exchange (ONNX) specification.
/// </summary>
/// <seealso href="https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.BFloat16.html"/>
[PublicAPI]
[DebuggerDisplay("{DebuggerDisplay}")]
[StructLayout(LayoutKind.Sequential)]
public readonly struct BFloat16 : IBinaryFloatingPointIeee754<BFloat16>,
    IMinMaxValue<BFloat16>
{
    private const ushort SignMask = 0x8000;
    private const ushort BiasedExponentMask = 0x7F80;
    private const int BiasedExponentShift = 0x07;
    private const byte ShiftedBiasedExponentMask = BiasedExponentMask >> BiasedExponentShift;
    private const ushort TrailingSignificandMask = 0x007F;
    private const byte MaxBiasedExponent = 0xFF;
    private const byte ExponentBias = 0x07F;
    private const ushort PositiveZeroBits = 0x0000;
    private const ushort NegativeZeroBits = 0x8000;
    private const ushort OneBits = 0x3F80;
    private const ushort NegativeOneBits = 0xBF80;
    private const ushort PositiveInfinityBits = 0x7F80;
    private const ushort NegativeInfinityBits = 0xFF80;
    private const ushort NaNBits = 0x7FC1;
    private const ushort MinValueBits = 0xFF7F;
    private const ushort MaxValueBits = 0x7F7F;
    private const ushort EpsilonBits = 0x0080;
    private const ushort PiBits = 0x4049;
    private const ushort EBits = 0x402E;
    private const ushort TauBits = 0x40C9;
    private const uint RoundingBase = 0x7FFF;
    private const uint SingleBiasedExponentMask = 0x7F80_0000;
    private const int SingleSignShift = 0x01F;
    private const uint SingleMostSignificantSigBit = 0x400000;

    private readonly ushort _value;

    private BFloat16(ushort value)
    {
        _value = value;
    }

    /// <inheritdoc />
    public static int Radix => 2;

    /// <inheritdoc />
    public static BFloat16 Zero => new (PositiveZeroBits);

    /// <inheritdoc />
    public static BFloat16 NegativeZero => new (NegativeZeroBits);

    /// <inheritdoc />
    public static BFloat16 One => new (OneBits);

    /// <inheritdoc />
    public static BFloat16 NegativeOne => new (NegativeOneBits);

    /// <inheritdoc />
    public static BFloat16 Epsilon => new (EpsilonBits);

    /// <inheritdoc />
    public static BFloat16 NaN => new (NaNBits);

    /// <inheritdoc />
    public static BFloat16 NegativeInfinity => new (NegativeInfinityBits);

    /// <inheritdoc />
    public static BFloat16 PositiveInfinity => new (PositiveInfinityBits);

    /// <inheritdoc />
    public static BFloat16 MaxValue => new (MaxValueBits);

    /// <inheritdoc />
    public static BFloat16 MinValue => new (MinValueBits);

    /// <inheritdoc />
    public static BFloat16 E => new (EBits);

    /// <inheritdoc />
    public static BFloat16 Pi => new (PiBits);

    /// <inheritdoc />
    public static BFloat16 Tau => new (TauBits);

    /// <inheritdoc />
    public static BFloat16 AdditiveIdentity => new (PositiveZeroBits);

    /// <inheritdoc />
    public static BFloat16 MultiplicativeIdentity => new (OneBits);

    [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
    [ExcludeFromCodeCoverage]
    private float DebuggerDisplay => (float)this;

    /// <summary>
    /// Implicitly converts a <see cref="BFloat16"/> to a <see cref="float"/>.
    /// </summary>
    /// <param name="value">The value to convert.</param>
    /// <returns>The converted value.</returns>
    public static implicit operator BFloat16(float value)
    {
        if (float.IsNaN(value))
        {
            return NaN;
        }

        if (float.IsPositiveInfinity(value))
        {
            return PositiveInfinity;
        }

        if (float.IsNegativeInfinity(value))
        {
            return NegativeInfinity;
        }

        var singleBits = SingleToUInt32Bits(value);
        var bfloatBits = SingleBitsToBFloat16Bits(singleBits);
        singleBits += ((uint)bfloatBits & 1) + RoundingBase;
        bfloatBits = SingleBitsToBFloat16Bits(singleBits);
        return new BFloat16(bfloatBits);
    }

    /// <summary>
    /// Explicitly converts a <see cref="BFloat16"/> to a <see cref="double"/>.
    /// </summary>
    /// <param name="value">The value to convert.</param>
    /// <returns>The converted value.</returns>
    public static explicit operator float(BFloat16 value)
    {
        var sign = IsNegative(value);
        int exp = value.GetBiasedExponent();
        uint sig = value.GetTrailingSignificand();

        switch (exp)
        {
            case MaxBiasedExponent when sig != 0:
                return CreateSingleNaN(sign, (ulong)sig << 56);
            case MaxBiasedExponent:
                return sign ? float.NegativeInfinity : float.PositiveInfinity;
            case 0 when sig == 0:
                return sign ? -0.0f : 0.0f;
            default:
                var singleBits = BFloat16BitsToSingleBits(value._value);
                return UInt32BitsToSingle(singleBits);
        }
    }

    /// <inheritdoc />
    public static bool operator ==(BFloat16 left, BFloat16 right)
    {
        if (IsNaN(left) && IsNaN(right))
        {
            return true;
        }

        if (IsNaN(left) || IsNaN(right))
        {
            return false;
        }

        return left._value == right._value;
    }

    /// <inheritdoc />
    public static bool operator !=(BFloat16 left, BFloat16 right)
    {
        return !(left == right);
    }

    /// <inheritdoc />
    public static bool operator >(BFloat16 left, BFloat16 right)
    {
        return right < left;
    }

    /// <inheritdoc />
    public static bool operator >=(BFloat16 left, BFloat16 right)
    {
        return right <= left;
    }

    /// <inheritdoc />
    public static bool operator <(BFloat16 left, BFloat16 right)
    {
        if (IsNaN(left) || IsNaN(right))
        {
            return false;
        }

        var leftIsNegative = IsNegative(left);

        if (leftIsNegative != IsNegative(right))
        {
            return leftIsNegative && !AreZero(left, right);
        }

        return left._value != right._value && ((left._value < right._value) ^ leftIsNegative);
    }

    /// <inheritdoc />
    public static bool operator <=(BFloat16 left, BFloat16 right)
    {
        if (IsNaN(left) || IsNaN(right))
        {
            return false;
        }

        var leftIsNegative = IsNegative(left);

        if (leftIsNegative != IsNegative(right))
        {
            return leftIsNegative || AreZero(left, right);
        }

        return left._value == right._value || ((left._value < right._value) ^ leftIsNegative);
    }

    /// <inheritdoc />
    public static BFloat16 operator ++(BFloat16 value)
    {
        return value + One;
    }

    /// <inheritdoc />
    public static BFloat16 operator --(BFloat16 value)
    {
        return value - One;
    }

    /// <inheritdoc />
    public static BFloat16 operator *(BFloat16 left, BFloat16 right)
    {
        return (float)left * (float)right;
    }

    /// <inheritdoc />
    public static BFloat16 operator /(BFloat16 left, BFloat16 right)
    {
        return (float)left / (float)right;
    }

    /// <inheritdoc />
    public static BFloat16 operator %(BFloat16 left, BFloat16 right)
    {
        return (float)left % (float)right;
    }

    /// <inheritdoc />
    public static BFloat16 operator +(BFloat16 left, BFloat16 right)
    {
        return (float)left + (float)right;
    }

    /// <inheritdoc />
    public static BFloat16 operator -(BFloat16 left, BFloat16 right)
    {
        return (float)left - (float)right;
    }

    /// <inheritdoc />
    public static BFloat16 operator -(BFloat16 value)
    {
        return (float)value * -1;
    }

    /// <inheritdoc />
    public static BFloat16 operator +(BFloat16 value)
    {
        return Abs(value);
    }

    /// <inheritdoc />
    public static BFloat16 operator &(BFloat16 left, BFloat16 right)
    {
        return left._value & right._value;
    }

    /// <inheritdoc />
    public static BFloat16 operator |(BFloat16 left, BFloat16 right)
    {
        return left._value | right._value;
    }

    /// <inheritdoc />
    public static BFloat16 operator ^(BFloat16 left, BFloat16 right)
    {
        return left._value ^ right._value;
    }

    /// <inheritdoc />
    public static BFloat16 operator ~(BFloat16 value)
    {
        return ~value._value;
    }

    /// <inheritdoc />
    public static BFloat16 Pow(BFloat16 x, BFloat16 y)
    {
        return (BFloat16)Math.Pow((float)x, (float)y);
    }

    /// <inheritdoc />
    public static BFloat16 Parse(string s, IFormatProvider? provider)
    {
        return float.Parse(s, provider);
    }

    /// <inheritdoc />
    public static BFloat16 Parse(ReadOnlySpan<char> s, NumberStyles style, IFormatProvider? provider)
    {
        return float.Parse(s, style, provider);
    }

    /// <inheritdoc />
    public static BFloat16 Parse(string s, NumberStyles style, IFormatProvider? provider)
    {
        return float.Parse(s, style, provider);
    }

    /// <inheritdoc />
    public static BFloat16 Parse(ReadOnlySpan<char> s, IFormatProvider? provider)
    {
        return float.Parse(s, provider);
    }

    /// <inheritdoc />
    public static bool TryParse(string? s, IFormatProvider? provider, out BFloat16 result)
    {
        if (float.TryParse(s, provider, out var f))
        {
            result = f;
            return true;
        }

        result = default;
        return false;
    }

    /// <inheritdoc />
    public static bool TryParse(ReadOnlySpan<char> s, IFormatProvider? provider, out BFloat16 result)
    {
        if (float.TryParse(s, provider, out var f))
        {
            result = f;
            return true;
        }

        result = default;
        return false;
    }

    /// <inheritdoc />
    public static bool TryParse(ReadOnlySpan<char> s, NumberStyles style, IFormatProvider? provider, out BFloat16 result)
    {
        var success = float.TryParse(
            s,
            style,
            provider,
            out var resultFloat);

        if (success)
        {
            result = resultFloat;
            return true;
        }

        result = default;
        return false;
    }

    /// <inheritdoc />
    public static bool TryParse(string? s, NumberStyles style, IFormatProvider? provider, out BFloat16 result)
    {
        var success = float.TryParse(
            s,
            style,
            provider,
            out var resultFloat);

        if (success)
        {
            result = resultFloat;
            return true;
        }

        result = default;
        return false;
    }

    /// <inheritdoc />
    public static BFloat16 Abs(BFloat16 value)
    {
        return new BFloat16(StripSign(value._value));
    }

    /// <inheritdoc />
    public static bool IsCanonical(BFloat16 value)
    {
        return true;
    }

    /// <inheritdoc />
    public static bool IsComplexNumber(BFloat16 value)
    {
        return false;
    }

    /// <inheritdoc />
    public static bool IsEvenInteger(BFloat16 value)
    {
        return float.IsEvenInteger((float)value);
    }

    /// <inheritdoc />
    public static bool IsFinite(BFloat16 value)
    {
        return StripSign(value) < PositiveInfinityBits;
    }

    /// <inheritdoc />
    public static bool IsImaginaryNumber(BFloat16 value)
    {
        return false;
    }

    /// <inheritdoc />
    public static bool IsInfinity(BFloat16 value)
    {
        return StripSign(value) == PositiveInfinityBits;
    }

    /// <inheritdoc />
    public static bool IsInteger(BFloat16 value)
    {
        return float.IsInteger((float)value);
    }

    /// <inheritdoc />
    public static bool IsNaN(BFloat16 value)
    {
        return StripSign(value) > PositiveInfinityBits;
    }

    /// <inheritdoc />
    public static bool IsNegative(BFloat16 value)
    {
        return (short)value._value < 0;
    }

    /// <inheritdoc />
    public static bool IsNegativeInfinity(BFloat16 value)
    {
        return value._value == NegativeInfinityBits;
    }

    /// <inheritdoc />
    public static bool IsNormal(BFloat16 value)
    {
        uint absValue = StripSign(value);
        return absValue < PositiveInfinityBits &&
               absValue != 0 &&
               (absValue & BiasedExponentMask) != 0;
    }

    /// <inheritdoc />
    public static bool IsOddInteger(BFloat16 value)
    {
        return float.IsOddInteger((float)value);
    }

    /// <inheritdoc />
    public static bool IsPositive(BFloat16 value)
    {
        return (short)value._value >= 0;
    }

    /// <inheritdoc />
    public static bool IsPositiveInfinity(BFloat16 value)
    {
        return value._value == PositiveInfinityBits;
    }

    /// <inheritdoc />
    public static bool IsRealNumber(BFloat16 value)
    {
#pragma warning disable CS1718 // Comparison made to same variable
        // ReSharper disable once EqualExpressionComparison
        return value == value;
#pragma warning restore CS1718 // Comparison made to same variable
    }

    /// <inheritdoc />
    public static bool IsSubnormal(BFloat16 value)
    {
        uint absValue = StripSign(value);
        return absValue < PositiveInfinityBits &&
               absValue != 0 &&
               (absValue & BiasedExponentMask) == 0;
    }

    /// <inheritdoc />
    public static bool IsZero(BFloat16 value)
    {
        return StripSign(value) == 0;
    }

    /// <inheritdoc />
    public static BFloat16 MaxMagnitude(BFloat16 x, BFloat16 y)
    {
        return Abs(x) > Abs(y) ? x : y;
    }

    /// <inheritdoc />
    public static BFloat16 MaxMagnitudeNumber(BFloat16 x, BFloat16 y)
    {
        return Abs(x) > Abs(y) ? x : y;
    }

    /// <inheritdoc />
    public static BFloat16 MinMagnitude(BFloat16 x, BFloat16 y)
    {
        return Abs(x) < Abs(y) ? x : y;
    }

    /// <inheritdoc />
    public static BFloat16 MinMagnitudeNumber(BFloat16 x, BFloat16 y)
    {
        return Abs(x) < Abs(y) ? x : y;
    }

    /// <inheritdoc />
    public static BFloat16 CreateChecked<TOther>(TOther value)
        where TOther : INumberBase<TOther>
    {
        return float.CreateChecked(value);
    }

    /// <inheritdoc />
    public static BFloat16 CreateSaturating<TOther>(TOther value)
        where TOther : INumberBase<TOther>
    {
        return float.CreateSaturating(value);
    }

    /// <inheritdoc />
    public static bool TryConvertFromChecked<TOther>(TOther value, out BFloat16 result)
        where TOther : INumberBase<TOther>
    {
        var success = TOther.TryConvertToChecked(value, out float floatValue);

        if (success)
        {
            result = floatValue;
            return true;
        }

        result = default;
        return false;
    }

    /// <inheritdoc />
    public static bool TryConvertFromSaturating<TOther>(TOther value, out BFloat16 result)
        where TOther : INumberBase<TOther>
    {
        var success = TOther.TryConvertToSaturating(value, out float floatValue);

        if (success)
        {
            result = floatValue;
            return true;
        }

        result = default;
        return false;
    }

    /// <inheritdoc />
    public static bool TryConvertFromTruncating<TOther>(TOther value, out BFloat16 result)
        where TOther : INumberBase<TOther>
    {
        var success = TOther.TryConvertToTruncating(value, out float floatValue);

        if (success)
        {
            result = floatValue;
            return true;
        }

        result = default;
        return false;
    }

    /// <inheritdoc />
    public static bool TryConvertToChecked<TOther>(BFloat16 value, out TOther result)
        where TOther : INumberBase<TOther>
    {
        return TOther.TryConvertFromChecked((float)value, out result!);
    }

    /// <inheritdoc />
    public static bool TryConvertToSaturating<TOther>(BFloat16 value, out TOther result)
        where TOther : INumberBase<TOther>
    {
        return TOther.TryConvertFromSaturating((float)value, out result!);
    }

    /// <inheritdoc />
    public static bool TryConvertToTruncating<TOther>(BFloat16 value, out TOther result)
        where TOther : INumberBase<TOther>
    {
        return TOther.TryConvertFromTruncating((float)value, out result!);
    }

    /// <inheritdoc />
    public static bool IsPow2(BFloat16 value)
    {
        return float.IsPow2((float)value);
    }

    /// <inheritdoc cref="IBinaryNumber{TSelf}.Log2" />
    public static BFloat16 Log2(BFloat16 value)
    {
        return float.Log2((float)value);
    }

    /// <inheritdoc />
    public static BFloat16 Log(BFloat16 x)
    {
        return float.Log((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Log(BFloat16 x, BFloat16 newBase)
    {
        return float.Log((float)x, (float)newBase);
    }

    /// <inheritdoc />
    public static BFloat16 Log10(BFloat16 x)
    {
        return float.Log10((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Exp(BFloat16 x)
    {
        return float.Exp((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Exp10(BFloat16 x)
    {
        return float.Exp10((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Exp2(BFloat16 x)
    {
        return float.Exp2((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Acosh(BFloat16 x)
    {
        return float.Acosh((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Asinh(BFloat16 x)
    {
        return float.Asinh((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Atanh(BFloat16 x)
    {
        return float.Atanh((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Cosh(BFloat16 x)
    {
        return float.Cosh((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Sinh(BFloat16 x)
    {
        return float.Sinh((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Tanh(BFloat16 x)
    {
        return float.Tanh((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Cbrt(BFloat16 x)
    {
        return float.Cbrt((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Hypot(BFloat16 x, BFloat16 y)
    {
        return float.Hypot((float)x, (float)y);
    }

    /// <inheritdoc />
    public static BFloat16 RootN(BFloat16 x, int n)
    {
        return float.RootN((float)x, n);
    }

    /// <inheritdoc />
    public static BFloat16 Sqrt(BFloat16 x)
    {
        return float.Sqrt((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Acos(BFloat16 x)
    {
        return float.Acos((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 AcosPi(BFloat16 x)
    {
        return float.AcosPi((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Asin(BFloat16 x)
    {
        return float.Asin((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 AsinPi(BFloat16 x)
    {
        return float.AsinPi((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Atan(BFloat16 x)
    {
        return float.Atan((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 AtanPi(BFloat16 x)
    {
        return float.AtanPi((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Cos(BFloat16 x)
    {
        return float.Cos((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 CosPi(BFloat16 x)
    {
        return float.CosPi((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Sin(BFloat16 x)
    {
        return float.Sin((float)x);
    }

    /// <inheritdoc />
    public static (BFloat16 Sin, BFloat16 Cos) SinCos(BFloat16 x)
    {
        return float.SinCos((float)x);
    }

    /// <inheritdoc />
    public static (BFloat16 SinPi, BFloat16 CosPi) SinCosPi(BFloat16 x)
    {
        return float.SinCosPi((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 SinPi(BFloat16 x)
    {
        return float.SinPi((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Tan(BFloat16 x)
    {
        return float.Tan((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 TanPi(BFloat16 x)
    {
        return float.TanPi((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Atan2(BFloat16 y, BFloat16 x)
    {
        return float.Atan2((float)y, (float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Atan2Pi(BFloat16 y, BFloat16 x)
    {
        return float.Atan2Pi((float)y, (float)x);
    }

    /// <inheritdoc />
    public static BFloat16 BitDecrement(BFloat16 x)
    {
        var bits = x._value;

        if ((bits & PositiveInfinityBits) >= PositiveInfinityBits)
        {
            return bits == PositiveInfinityBits ? MaxValue : x;
        }

        if (bits == PositiveZeroBits)
        {
            return -Epsilon;
        }

        bits += (ushort)((short)bits < 0 ? +1 : -1);
        return new BFloat16(bits);
    }

    /// <inheritdoc />
    public static BFloat16 BitIncrement(BFloat16 x)
    {
        var bits = x._value;

        if ((bits & PositiveInfinityBits) >= PositiveInfinityBits)
        {
            return (bits == NegativeInfinityBits) ? MinValue : x;
        }

        if (bits == NegativeZeroBits)
        {
            return Epsilon;
        }

        bits += (ushort)((short)bits < 0 ? -1 : +1);
        return new BFloat16(bits);
    }

    /// <inheritdoc />
    public static BFloat16 FusedMultiplyAdd(BFloat16 left, BFloat16 right, BFloat16 addend)
    {
        return float.FusedMultiplyAdd((float)left, (float)right, (float)addend);
    }

    /// <inheritdoc />
    public static BFloat16 Ieee754Remainder(BFloat16 left, BFloat16 right)
    {
        return float.Ieee754Remainder((float)left, (float)right);
    }

    /// <inheritdoc />
    public static int ILogB(BFloat16 x)
    {
        return float.ILogB((float)x);
    }

    /// <inheritdoc />
    public static BFloat16 Round(BFloat16 x, int digits, MidpointRounding mode)
    {
        return float.Round((float)x, digits, mode);
    }

    /// <inheritdoc />
    public static BFloat16 ScaleB(BFloat16 x, int n)
    {
        return float.ScaleB((float)x, n);
    }

    /// <inheritdoc />
    public string ToString(string? format, IFormatProvider? formatProvider)
    {
        return ((float)this).ToString(format, formatProvider);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return ((float)this).ToString(CultureInfo.CurrentCulture);
    }

    /// <inheritdoc />
    public bool TryFormat(Span<char> destination, out int charsWritten, ReadOnlySpan<char> format, IFormatProvider? provider)
    {
        return ((float)this).TryFormat(
            destination,
            out charsWritten,
            format,
            provider);
    }

    /// <inheritdoc />
    public int GetExponentByteCount()
    {
        return sizeof(sbyte);
    }

    /// <inheritdoc />
    public int GetExponentShortestBitLength()
    {
        var exponent = (sbyte)(GetBiasedExponent() - ExponentBias);

        if (exponent >= 0)
        {
            return (sizeof(sbyte) * 8) - sbyte.LeadingZeroCount(exponent);
        }

        return (sizeof(sbyte) * 8) + 1 - sbyte.LeadingZeroCount((sbyte)~exponent);
    }

    /// <inheritdoc />
    public int GetSignificandBitLength()
    {
        return sizeof(ushort) * 8;
    }

    /// <inheritdoc />
    public int GetSignificandByteCount()
    {
        return sizeof(ushort);
    }

    /// <inheritdoc />
    public bool TryWriteExponentBigEndian(Span<byte> destination, out int bytesWritten)
    {
        if (destination.Length >= sizeof(sbyte))
        {
            var exponent = (sbyte)(GetBiasedExponent() - ExponentBias);
            Unsafe.WriteUnaligned(ref MemoryMarshal.GetReference(destination), exponent);

            bytesWritten = sizeof(sbyte);
            return true;
        }

        bytesWritten = 0;
        return false;
    }

    /// <inheritdoc />
    public bool TryWriteExponentLittleEndian(Span<byte> destination, out int bytesWritten)
    {
        if (destination.Length >= sizeof(sbyte))
        {
            var exponent = (sbyte)(GetBiasedExponent() - ExponentBias);
            Unsafe.WriteUnaligned(ref MemoryMarshal.GetReference(destination), exponent);

            bytesWritten = sizeof(sbyte);
            return true;
        }

        bytesWritten = 0;
        return false;
    }

    /// <inheritdoc />
    public bool TryWriteSignificandBigEndian(Span<byte> destination, out int bytesWritten)
    {
        if (destination.Length >= sizeof(ushort))
        {
            var significand = (ushort)(GetTrailingSignificand() | (GetBiasedExponent() != 0 ? 1U << BiasedExponentShift : 0U));

            if (BitConverter.IsLittleEndian)
            {
                significand = BinaryPrimitives.ReverseEndianness(significand);
            }

            Unsafe.WriteUnaligned(ref MemoryMarshal.GetReference(destination), significand);

            bytesWritten = sizeof(ushort);
            return true;
        }

        bytesWritten = 0;
        return false;
    }

    /// <inheritdoc />
    public bool TryWriteSignificandLittleEndian(Span<byte> destination, out int bytesWritten)
    {
        if (destination.Length >= sizeof(ushort))
        {
            var significand = (ushort)(GetTrailingSignificand() | (GetBiasedExponent() != 0 ? 1U << BiasedExponentShift : 0U));

            if (!BitConverter.IsLittleEndian)
            {
                significand = BinaryPrimitives.ReverseEndianness(significand);
            }

            Unsafe.WriteUnaligned(ref MemoryMarshal.GetReference(destination), significand);

            bytesWritten = sizeof(ushort);
            return true;
        }

        bytesWritten = 0;
        return false;
    }

    /// <inheritdoc />
    public int CompareTo(object? obj)
    {
        if (obj is BFloat16 bFloat16)
        {
            return CompareTo(bFloat16);
        }

        throw new ArgumentException($"Object must be of type {nameof(BFloat16)}.", nameof(obj));
    }

    /// <inheritdoc />
    public int CompareTo(BFloat16 other)
    {
        if (this < other)
        {
            return -1;
        }

        if (this > other)
        {
            return 1;
        }

        if (this == other)
        {
            return 0;
        }

        if (IsNaN(this))
        {
            return IsNaN(other) ? 0 : -1;
        }

        return 1;
    }

    /// <inheritdoc />
    public bool Equals(BFloat16 other)
    {
        return AreZero(this, other) || (IsNaN(this) && IsNaN(other)) || _value == other._value;
    }

    /// <inheritdoc />
    public override bool Equals(object? obj)
    {
        return obj is BFloat16 other && Equals(other);
    }

    /// <inheritdoc />
    public override int GetHashCode()
    {
        return _value.GetHashCode();
    }

    private static ushort StripSign(BFloat16 value)
    {
        return (ushort)(value._value & ~SignMask);
    }

    private static bool AreZero(BFloat16 bFloat16, BFloat16 other)
    {
        return (bFloat16._value == PositiveZeroBits && other._value == NegativeZeroBits) ||
               (bFloat16._value == NegativeZeroBits && other._value == PositiveZeroBits);
    }

    private static uint SingleToUInt32Bits(float single)
    {
        uint result;

        unsafe
        {
            Buffer.MemoryCopy(
                &single,
                &result,
                sizeof(uint),
                sizeof(uint));
        }

        return result;
    }

    private static float UInt32BitsToSingle(uint singleBits)
    {
        float result;

        unsafe
        {
            Buffer.MemoryCopy(
                &singleBits,
                &result,
                sizeof(uint),
                sizeof(uint));
        }

        return result;
    }

    private static ushort SingleBitsToBFloat16Bits(uint singleBits)
    {
        if (!BitConverter.IsLittleEndian)
        {
            return (ushort)(singleBits & 0xFFFF);
        }

        return (ushort)(singleBits >> 16);
    }

    private static uint BFloat16BitsToSingleBits(ushort bfloatBits)
    {
        if (!BitConverter.IsLittleEndian)
        {
            return bfloatBits;
        }

        return (uint)bfloatBits << 16;
    }

    private static float CreateSingleNaN(bool sign, ulong significand)
    {
        // We need to set at least on bit in NaN significant
        const uint naNBits = SingleBiasedExponentMask | SingleMostSignificantSigBit;

        var signInt = (sign ? 1U : 0U) << SingleSignShift;
        var sigInt = (uint)(significand >> 41);
        var singleBits = signInt | naNBits | sigInt;

        return UInt32BitsToSingle(singleBits);
    }

    private static byte ExtractBiasedExponentFromBits(ushort bits)
    {
        return (byte)((bits >> BiasedExponentShift) & ShiftedBiasedExponentMask);
    }

    private static ushort ExtractTrailingSignificandFromBits(ushort bits)
    {
        return (ushort)(bits & TrailingSignificandMask);
    }

    private byte GetBiasedExponent()
    {
        var bits = _value;
        return ExtractBiasedExponentFromBits(bits);
    }

    private ushort GetTrailingSignificand()
    {
        var bits = _value;
        return ExtractTrailingSignificandFromBits(bits);
    }
}