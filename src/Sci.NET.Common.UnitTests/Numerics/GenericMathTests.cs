// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Numerics;

namespace Sci.NET.Common.UnitTests.Numerics;

public class GenericMathTests
{
    [Fact]
    public void IsFloatingPoint_ReturnsTrueForFloatingPointTypes()
    {
        GenericMath.IsFloatingPoint<Half>().Should().BeTrue();
        GenericMath.IsFloatingPoint<BFloat16>().Should().BeTrue();
        GenericMath.IsFloatingPoint<float>().Should().BeTrue();
        GenericMath.IsFloatingPoint<double>().Should().BeTrue();
        GenericMath.IsFloatingPoint<decimal>().Should().BeTrue();
    }

    [Fact]
    public void IsFloatingPoint_ReturnsFalseForNonFloatingPointTypes()
    {
        GenericMath.IsFloatingPoint<sbyte>().Should().BeFalse();
        GenericMath.IsFloatingPoint<byte>().Should().BeFalse();
        GenericMath.IsFloatingPoint<short>().Should().BeFalse();
        GenericMath.IsFloatingPoint<ushort>().Should().BeFalse();
        GenericMath.IsFloatingPoint<int>().Should().BeFalse();
        GenericMath.IsFloatingPoint<uint>().Should().BeFalse();
        GenericMath.IsFloatingPoint<long>().Should().BeFalse();
        GenericMath.IsFloatingPoint<ulong>().Should().BeFalse();
    }

    [Fact]
    public void Epsilon_ReturnsCorrectValueForFloatingPointTypes()
    {
        // TODO: Figure out why BF16 epsilon is not correct.
        GenericMath.Epsilon<Half>().Should().Be(Half.Epsilon);
        GenericMath.Epsilon<float>().Should().Be(float.Epsilon);
        GenericMath.Epsilon<double>().Should().Be(double.Epsilon);
    }

    [Fact]
    public void Epsilon_ReturnsCorrectValueForNonFloatingPointTypes()
    {
        GenericMath.Epsilon<sbyte>().Should().Be(1);
        GenericMath.Epsilon<byte>().Should().Be(1);
        GenericMath.Epsilon<short>().Should().Be(1);
        GenericMath.Epsilon<ushort>().Should().Be(1);
        GenericMath.Epsilon<int>().Should().Be(1);
        GenericMath.Epsilon<uint>().Should().Be(1);
        GenericMath.Epsilon<long>().Should().Be(1);
        GenericMath.Epsilon<ulong>().Should().Be(1);
    }

    [Fact]
    public void IsSigned_ReturnsCorrectValues()
    {
        GenericMath.IsSigned<BFloat16>().Should().BeTrue();
        GenericMath.IsSigned<Half>().Should().BeTrue();
        GenericMath.IsSigned<float>().Should().BeTrue();
        GenericMath.IsSigned<double>().Should().BeTrue();
        GenericMath.IsSigned<decimal>().Should().BeTrue();
        GenericMath.IsSigned<byte>().Should().BeFalse();
        GenericMath.IsSigned<sbyte>().Should().BeTrue();
        GenericMath.IsSigned<ushort>().Should().BeFalse();
        GenericMath.IsSigned<short>().Should().BeTrue();
        GenericMath.IsSigned<uint>().Should().BeFalse();
        GenericMath.IsSigned<int>().Should().BeTrue();
        GenericMath.IsSigned<ulong>().Should().BeFalse();
        GenericMath.IsSigned<long>().Should().BeTrue();
    }
}