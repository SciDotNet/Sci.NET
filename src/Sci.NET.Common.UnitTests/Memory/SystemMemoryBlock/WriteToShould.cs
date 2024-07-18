// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory.SystemMemoryBlock;

public class WriteToShould
{
    [Fact]
    public void WriteBytesCorrectly_GivenLength()
    {
        using var memoryBlock = new SystemMemoryBlock<byte>(32);
        using var destination = new MemoryStream();

        memoryBlock.Fill(0x10);

        memoryBlock.WriteTo(destination);

        var result = destination.ToArray();

        for (var i = 0; i < 32; i++)
        {
            result[i].Should().Be(0x10);
        }
    }

    [Fact]
    public void ThrowException_WhenDisposed()
    {
        var memoryBlock = new SystemMemoryBlock<int>(10);
        memoryBlock.Dispose();

        Action act = () => memoryBlock.WriteTo(new MemoryStream());

        act.Should().Throw<ObjectDisposedException>();
    }
}