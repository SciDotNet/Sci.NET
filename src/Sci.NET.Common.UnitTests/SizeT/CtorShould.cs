// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.UnitTests.SizeT;

public class CtorShould
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(int.MaxValue)]
    public void ReturnNewInstance_GivenValidInt32(int value)
    {
        var sizeT = new Common.SizeT(value);

        sizeT.Should().NotBeNull();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(uint.MaxValue)]
    public void ReturnNewInstance_GivenValidUInt32(uint value)
    {
        var sizeT = new Common.SizeT(value);

        sizeT.Should().NotBeNull();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(long.MaxValue)]
    public void ReturnNewInstance_GivenValidInt64(long value)
    {
        var sizeT = new Common.SizeT(value);

        sizeT.Should().NotBeNull();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(ulong.MaxValue)]
    public void ReturnNewInstance_GivenValidUInt64(ulong value)
    {
        var sizeT = new Common.SizeT(value);

        sizeT.Should().NotBeNull();
    }

    [Fact]
    public void ThrowArgumentOutOfRangeException_GivenNegativeInt32()
    {
        var act = () => new Common.SizeT(-1);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void ThrowOutOfRangeException_GivenNegativeNInt()
    {
        var act = () => new Common.SizeT(new IntPtr(-1));

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void ThrowArgumentOutOfRangeException_GivenNegativeInt64()
    {
        var act = () => new Common.SizeT(-1L);

        act.Should().NotThrow();
    }
}