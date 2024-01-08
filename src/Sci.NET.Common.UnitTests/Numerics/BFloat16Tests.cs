// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

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
    [InlineData(0.0f, 0x0000)]
    [InlineData(-0.0f, 0x8000)]
    [InlineData(1.0f, 0x3F80)]
    [InlineData(-1.0f, 0xBF80)]
    [InlineData(2.0f, 0x4000)]
    [InlineData(-2.0f, 0xC000)]
    [InlineData(float.PositiveInfinity, 0x7F80)]
    [InlineData(float.NegativeInfinity, 0xFF80)]
    [InlineData(float.NaN, 0x7FC1)]
    public void ExplicitAndImplicitCastFromFloat_ReturnsCorrectBits(float value, ushort expected)
    {
        BFloat16 bfloat16Implicit = value;
        var bfloat16Explicit = (BFloat16)value;

        bfloat16Implicit.ReinterpretCast<BFloat16, ushort>().Should().Be(expected);
        bfloat16Explicit.ReinterpretCast<BFloat16, ushort>().Should().Be(expected);
    }

    [Theory]
    [InlineData(0.0f, 0x0000)]
    [InlineData(-0.0f, 0x8000)]
    [InlineData(1.0f, 0x3F80)]
    [InlineData(-1.0f, 0xBF80)]
    [InlineData(2.0f, 0x4000)]
    [InlineData(-2.0f, 0xC000)]
    [InlineData(float.PositiveInfinity, 0x7F80)]
    [InlineData(float.NegativeInfinity, 0xFF80)]
    [InlineData(float.NaN, 0x7FC1)]
    public void ExplicitCastToFloat_ReturnsCorrectBits(float expected, ushort bfloatValue)
    {
        var bfloat16 = bfloatValue.ReinterpretCast<ushort, BFloat16>();
        var float16Implicit = (float)bfloat16;

        float16Implicit.Should().Be(expected);
    }

#pragma warning disable xUnit1025
    [Theory]
    [InlineData(0.0f, 0.0f, true)]
    [InlineData(-0.0f, -0.0f, true)]
    [InlineData(1.0f, 1.0f, true)]
    [InlineData(-1.0f, -1.0f, true)]
    [InlineData(2.0f, 1.0f, false)]
    [InlineData(-2.0f, -1.0f, false)]
    [InlineData(float.PositiveInfinity, float.PositiveInfinity, true)]
    [InlineData(float.NegativeInfinity, float.NegativeInfinity, true)]
    [InlineData(float.NaN, float.NaN, true)]
#pragma warning restore xUnit1025
    public void EqualsAndEqualityOperator_ReturnsExpectedResult(BFloat16 left, BFloat16 right, bool expected)
    {
        left.Equals(right).Should().Be(expected);
        (left == right).Should().Be(expected);
        (left != right).Should().Be(!expected);
    }

#pragma warning disable xUnit1025
    [Theory]
    [InlineData(0.0f, 0.0f, false)]
    [InlineData(-0.0f, 0.0f, false)]
    [InlineData(1.0f, 0.9f, false)]
    [InlineData(-1.0f, -0.9f, true)]
    [InlineData(2.0f, 1.0f, false)]
    [InlineData(float.NaN, float.NaN, false)]
#pragma warning restore xUnit1025
    public void LessThanOperator_ReturnsExpectedResult(BFloat16 left, BFloat16 right, bool expected)
    {
        (left < right).Should().Be(expected);
    }

#pragma warning disable xUnit1025
    [Theory]
    [InlineData(0.0f, 0.0f, true)]
    [InlineData(0.0f, -0.0f, true)]
    [InlineData(1.0f, 0.9f, false)]
    [InlineData(-1.0f, -0.9f, true)]
    [InlineData(2.0f, 1.0f, false)]
    [InlineData(float.NaN, float.NaN, false)]
#pragma warning restore xUnit1025
    public void LessThanOrEqualOperator_ReturnsExpectedResult(BFloat16 left, BFloat16 right, bool expected)
    {
        (left <= right).Should().Be(expected);
    }
}