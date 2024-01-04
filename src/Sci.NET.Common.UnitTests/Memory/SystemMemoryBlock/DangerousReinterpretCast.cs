// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory.SystemMemoryBlock;

public class DangerousReinterpretCast
{
    [Fact]
    public void CastCorrectly_GivenInt()
    {
        var data = new int[] { -1, -2, -3, -4 };
        var block = new SystemMemoryBlock<int>(4);

        block.CopyFrom(data);

        var result = block.DangerousReinterpretCast<uint>();

        result[0].Should().Be(uint.MaxValue);
        result[1].Should().Be(uint.MaxValue - 1);
        result[2].Should().Be(uint.MaxValue - 2);
        result[3].Should().Be(uint.MaxValue - 3);
    }
}