// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory.SystemMemoryBlock;

public class FillShould
{
    [Theory]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(2048)]
    [InlineData(2049)]
    [InlineData(4096)]
    [InlineData(4097)]
    public void FillBytesCorrectly_GivenLength(int length)
    {
        var memoryBlock = new SystemMemoryBlock<byte>(length);

        memoryBlock.Fill(0x42);

        for (var i = 0; i < length; i++)
        {
            memoryBlock[i].Should().Be(0x42);
        }
    }

    [Theory]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(2048)]
    [InlineData(2049)]
    [InlineData(4096)]
    [InlineData(4097)]
    public void FillShortCorrectly_GivenLength(int length)
    {
        var memoryBlock = new SystemMemoryBlock<short>(length);

        memoryBlock.Fill(0x42);

        for (var i = 0; i < length; i++)
        {
            memoryBlock[i].Should().Be(0x42);
        }
    }

    [Theory]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(2048)]
    [InlineData(2049)]
    [InlineData(4096)]
    [InlineData(4097)]
    public void FillIntsCorrectly_GivenLength(int length)
    {
        var memoryBlock = new SystemMemoryBlock<int>(length);

        memoryBlock.Fill(0x42);

        for (var i = 0; i < length; i++)
        {
            memoryBlock[i].Should().Be(0x42);
        }
    }

    [Theory]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(2048)]
    [InlineData(2049)]
    [InlineData(4096)]
    [InlineData(4097)]
    public void FillLongCorrectly_GivenLength(int length)
    {
        var memoryBlock = new SystemMemoryBlock<long>(length);

        memoryBlock.Fill(0x42);

        for (var i = 0; i < length; i++)
        {
            memoryBlock[i].Should().Be(0x42);
        }
    }

    [Theory]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(2048)]
    [InlineData(2049)]
    [InlineData(4096)]
    [InlineData(4097)]
    public void FillGuidCorrectly_GivenLength(int length)
    {
        var memoryBlock = new SystemMemoryBlock<Guid>(length);

        var value = Guid.NewGuid();

        memoryBlock.Fill(value);

        for (var i = 0; i < length; i++)
        {
            memoryBlock[i].Should().Be(value);
        }
    }
}