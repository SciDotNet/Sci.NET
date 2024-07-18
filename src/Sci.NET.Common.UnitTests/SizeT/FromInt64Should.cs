// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.UnitTests.SizeT;

public class FromInt64Should
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(long.MaxValue)]
    public void ReturnNewInstance_GivenValidInt64(long value)
    {
        var sizeT = Common.SizeT.FromInt64(value);

        sizeT.Should().NotBeNull();
    }
}