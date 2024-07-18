// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.UnitTests.SizeT;

public class ToUIntPtrShould
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(int.MaxValue)]
    public void ReturnUIntPtr_GivenValidInt32(int value)
    {
        var sizeT = new Common.SizeT(value);

        var uIntPtr = sizeT.ToUIntPtr();

        uIntPtr.Should().Be((nuint)value);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(uint.MaxValue)]
    public void ReturnUIntPtr_GivenValidUInt32(uint value)
    {
        var sizeT = new Common.SizeT(value);

        var uIntPtr = sizeT.ToUIntPtr();

        uIntPtr.Should().Be(value);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(long.MaxValue)]
    public void ReturnUIntPtr_GivenValidInt64(long value)
    {
        var sizeT = new Common.SizeT(value);

        var uIntPtr = sizeT.ToUIntPtr();

        uIntPtr.Should().Be((nuint)value);
    }
}