// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory.SystemMemoryBlock;

public class CopyFrom
{
    [Theory]
    [InlineData(32)]
    [InlineData(33)]
    public void CopyBytesCorrectly_GivenLength(int count)
    {
        var memoryBlock = new SystemMemoryBlock<byte>(count);
        var source = Enumerable.Range(0, count).Select(x => (byte)x).ToArray();

        memoryBlock.CopyFrom(source);

        for (var i = 0; i < count; i++)
        {
            memoryBlock[i].Should().Be((byte)i);
        }
    }

    [Fact]
    public void ThrowArgumentException_GivenDifferentLength()
    {
        var memoryBlock = new SystemMemoryBlock<byte>(32);
        var source = new byte[33];

        var act = () => memoryBlock.CopyFrom(source);

        act.Should().Throw<ArgumentException>();
    }
}