// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.LowLevel;

namespace Sci.NET.Common.UnitTests.LowLevel;

public class CastingExtensionsTests
{
    [Fact]
    public void ReinterpretCast_Casts_IfTypeSizesAreSameLength()
    {
        const int value = 1;
        var cast = value.ReinterpretCast<int, uint>();

        cast.Should().Be(value);
    }

    [Fact]
    public void ReinterpretCast_Throws_IfTypeSizesAreDifferentLength()
    {
        var act = () => 1.ReinterpretCast<int, long>();

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void ReinterpretCast_Casts_HandlesNegativeAppropriately()
    {
        const int value = -1;
        var cast = value.ReinterpretCast<int, uint>();

        cast.Should().Be(uint.MaxValue);
    }
}