// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Sci.NET.Common.LowLevel;
using Sci.NET.Common.Numerics;

namespace Sci.NET.Common.UnitTests.Numerics;

public class BFloat16Tests
{
    [Fact]
    public void GetZero_ReturnsCorrectBits()
    {
        var zero = BFloat16.Zero;

        zero.ReinterpretCast<BFloat16, ushort>().Should().Be(0x0000);
    }

    [Fact]
    public void GetNegativeZero_ReturnsCorrectBits()
    {
        var negativeZero = BFloat16.NegativeZero;

        negativeZero.ReinterpretCast<BFloat16, ushort>().Should().Be(0x8000);
    }

    [Fact]
    public void GetOne_ReturnsCorrectBits()
    {
        var one = BFloat16.One;

        one.ReinterpretCast<BFloat16, ushort>().Should().Be(0x3F80);
    }

    [Fact]
    public void GetNegativeOne_ReturnsCorrectBits()
    {
        var negativeOne = BFloat16.NegativeOne;

        negativeOne.ReinterpretCast<BFloat16, ushort>().Should().Be(0xBF80);
    }

    [Fact]
    public void GetRadix_ReturnsCorrectValue()
    {
        var radix = BFloat16.Radix;

        radix.Should().Be(2);
    }

    [Fact]
    public void GetEpsilon_ReturnsCorrectValue()
    {
        var epsilon = BFloat16.Epsilon;

        epsilon.ReinterpretCast<BFloat16, ushort>().Should().Be(0x0080);
    }

    [Fact]
    public void GetPositiveInfinity_ReturnsCorrectBits()
    {
        var infinity = BFloat16.PositiveInfinity;

        infinity.ReinterpretCast<BFloat16, ushort>().Should().Be(0x7F80);
    }

    [Fact]
    public void GetNegativeInfinity_ReturnsCorrectBits()
    {
        var infinity = BFloat16.NegativeInfinity;

        infinity.ReinterpretCast<BFloat16, ushort>().Should().Be(0xFF80);
    }

    [Fact]
    public void GetNaN_ReturnsCorrectBits()
    {
        var nan = BFloat16.NaN;

        nan.ReinterpretCast<BFloat16, ushort>().Should().Be(0x7FC1);
    }

    [Fact]
    public void MaxValue_ReturnsCorrectBits()
    {
        var max = BFloat16.MaxValue;

        max.ReinterpretCast<BFloat16, ushort>().Should().Be(0x7F7F);
    }

    [Fact]
    public void MinValue_ReturnsCorrectBits()
    {
        var min = BFloat16.MinValue;

        min.ReinterpretCast<BFloat16, ushort>().Should().Be(0xFF7F);
    }

    [Fact]
    public void E_ReturnsCorrectBits()
    {
        var e = BFloat16.E;

        e.ReinterpretCast<BFloat16, ushort>().Should().Be(0x402E);
    }

    [Fact]
    public void Pi_ReturnsCorrectBits()
    {
        var pi = BFloat16.Pi;

        pi.ReinterpretCast<BFloat16, ushort>().Should().Be(0x4049);
    }

    [Fact]
    public void Tau_ReturnsCorrectBits()
    {
        var tau = BFloat16.Tau;

        tau.ReinterpretCast<BFloat16, ushort>().Should().Be(0x40C9);
    }

    [Fact]
    public void AdditiveIdentity_ReturnsCorrectBits()
    {
        var additiveIdentity = BFloat16.AdditiveIdentity;

        additiveIdentity.ReinterpretCast<BFloat16, ushort>().Should().Be(0x0000);
    }

    [Fact]
    public void MultiplicativeIdentity_ReturnsCorrectBits()
    {
        var multiplicativeIdentity = BFloat16.MultiplicativeIdentity;

        multiplicativeIdentity.ReinterpretCast<BFloat16, ushort>().Should().Be(0x3F80);
    }

    [Theory]
    [InlineData(10.0f, 5.0f, true)]
    [InlineData(5.0f, 10.0f, false)]
    [InlineData(5.0f, 5.0f, false)]
    public void GreaterThanOperator_ReturnsCorrectValue(BFloat16 left, BFloat16 right, bool expected)
    {
        var result = left > right;

        result.Should().Be(expected);
    }

    [Theory]
    [InlineData(10.0f, 5.0f, false)]
    [InlineData(5.0f, 10.0f, true)]
    [InlineData(5.0f, 5.0f, false)]
    public void LessThanOperator_ReturnsCorrectValue(BFloat16 left, BFloat16 right, bool expected)
    {
        var result = left < right;

        result.Should().Be(expected);
    }

    [Theory]
    [InlineData(10.0f, 5.0f, true)]
    [InlineData(5.0f, 10.0f, false)]
    [InlineData(5.0f, 5.0f, true)]
    public void GreaterThanOrEqualOperator_ReturnsCorrectValue(BFloat16 left, BFloat16 right, bool expected)
    {
        var result = left >= right;

        result.Should().Be(expected);
    }

    [Theory]
    [InlineData(10.0f, 5.0f, false)]
    [InlineData(5.0f, 10.0f, true)]
    [InlineData(5.0f, 5.0f, true)]
    public void LessThanOrEqualOperator_ReturnsCorrectValue(BFloat16 left, BFloat16 right, bool expected)
    {
        var result = left <= right;

        result.Should().Be(expected);
    }

    [Theory]
    [InlineData(10.0f, 11.0f)]
    [InlineData(5.0f, 6.0f)]
    [InlineData(-10.0f, -9.0f)]
    public void IncrementOperator_ReturnsCorrectValue(BFloat16 value, BFloat16 expected)
    {
        value++;

        value.Should().Be(expected);
    }

    [Theory]
    [InlineData(10.0f, 9.0f)]
    [InlineData(5.0f, 4.0f)]
    [InlineData(-10.0f, -11.0f)]
    public void DecrementOperator_ReturnsCorrectValue(BFloat16 value, BFloat16 expected)
    {
        value--;

        value.Should().Be(expected);
    }

    [Theory]
    [InlineData("10.0", 10.0f)]
    [InlineData("5.0", 5.0f)]
    [InlineData("-10.0", -10.0f)]
    [InlineData("10", 10.0f)]
    [InlineData("-1", -1.0f)]
    [InlineData("3.14159265358979323846264338327950288419716", 3.140625f)]
    [SuppressMessage("Globalization", "CA1305:Specify IFormatProvider", Justification = "Testing for invariant culture.")]
    public void Parse_ReturnsCorrectValue(string value, BFloat16 expected)
    {
        var result = BFloat16.Parse(value);

        result.Should().Be(expected);
    }
}