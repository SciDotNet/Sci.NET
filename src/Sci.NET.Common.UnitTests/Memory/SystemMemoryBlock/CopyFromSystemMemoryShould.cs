// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory.SystemMemoryBlock;

public class CopyFromSystemMemoryShould
{
    [Theory]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(2048)]
    [InlineData(2049)]
    public void CopyBytesCorrectly_GivenLength(int count)
    {
        var memoryBlock = new SystemMemoryBlock<byte>(count);
        var destination = new SystemMemoryBlock<byte>(count);

        memoryBlock.Fill(0x42);

        destination.CopyFromSystemMemory(memoryBlock);

        for (var i = 0; i < count; i++)
        {
            destination[i].Should().Be(0x42);
        }
    }

    [Fact]
    public void ThrowArgumentException_GivenDifferentLength()
    {
        var memoryBlock = new SystemMemoryBlock<byte>(32);
        var destination = new SystemMemoryBlock<byte>(33);

        memoryBlock.Fill(0x42);

        var act = () => destination.CopyFromSystemMemory(memoryBlock);

        act.Should().Throw<ArgumentException>();
    }
}