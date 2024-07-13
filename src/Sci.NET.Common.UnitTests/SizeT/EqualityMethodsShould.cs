// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.UnitTests.SizeT;

public class EqualityMethodsShould
{
    [Fact]
    public void ReturnTrue_GivenEqualValues()
    {
        var sizeT1 = new Common.SizeT(1);
        var sizeT2 = new Common.SizeT(1);

        var result = sizeT1.Equals(sizeT2);

        result.Should().BeTrue();
    }

    [Fact]
    public void ReturnFalse_GivenDifferentValues()
    {
        var sizeT1 = new Common.SizeT(1);
        var sizeT2 = new Common.SizeT(2);

        var result = sizeT1.Equals(sizeT2);

        result.Should().BeFalse();
    }

    [Fact]
    public void ReturnFalse_GivenNull()
    {
        var sizeT = new Common.SizeT(1);

        var result = sizeT.Equals(null);

        result.Should().BeFalse();
    }

    [Fact]
    public void ReturnTrue_GivenBoxedEqualValue()
    {
        var sizeT1 = new Common.SizeT(1);
        var sizeT2 = new Common.SizeT(1);

        var result = sizeT1.Equals((object)sizeT2);

        result.Should().BeTrue();
    }

    [Fact]
    public void ReturnFalse_GivenBoxedDifferentValue()
    {
        var sizeT1 = new Common.SizeT(1);
        var sizeT2 = new Common.SizeT(2);

        var result = sizeT1.Equals((object)sizeT2);

        result.Should().BeFalse();
    }

    [Fact]
    public void ReturnFalse_GivenDifferentObject()
    {
        var sizeT = new Common.SizeT(1);

        var result = sizeT.Equals(new object());

        result.Should().BeFalse();
    }
}