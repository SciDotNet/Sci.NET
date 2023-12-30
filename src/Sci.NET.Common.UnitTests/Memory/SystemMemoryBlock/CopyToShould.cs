// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory.SystemMemoryBlock;

public class CopyToShould
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

        memoryBlock.CopyTo(destination);

        for (var i = 0; i < count; i++)
        {
            destination[i].Should().Be(0x42);
        }
    }
}