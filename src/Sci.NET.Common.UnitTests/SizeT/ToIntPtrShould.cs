// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.UnitTests.SizeT;

public class ToIntPtrShould
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(int.MaxValue)]
    public void ReturnIntPtr_GivenValidInt32(int value)
    {
        var sizeT = new Common.SizeT(value);

        var intPtr = sizeT.ToIntPtr();

        intPtr.Should().Be(value);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(uint.MaxValue)]
    public void ReturnIntPtr_GivenValidUInt32(uint value)
    {
        var sizeT = new Common.SizeT(value);

        var intPtr = sizeT.ToIntPtr();

        intPtr.Should().Be((IntPtr)value);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(long.MaxValue)]
    public void ReturnIntPtr_GivenValidInt64(long value)
    {
        var sizeT = new Common.SizeT(value);

        var intPtr = sizeT.ToIntPtr();

        intPtr.Should().Be(new IntPtr(value));
    }
}