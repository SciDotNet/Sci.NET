// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory.SystemMemoryBlock;

public class AsSpanShould
{
    [Fact]
    public void ReturnCorrectSpan_GivenValidInstance()
    {
        var data = new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var memoryBlock = new SystemMemoryBlock<int>(10);
        memoryBlock.CopyFrom(data);

        var span = memoryBlock.AsSpan();

        span.Length.Should().Be((int)memoryBlock.Length);

        for (var i = 0; i < span.Length; i++)
        {
            span[i].Should().Be(data[i]);
        }
    }

    [Fact]
    public void ReturnCorrectSpan_GivenValidInstanceAndSubRange()
    {
        var data = new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        using var memoryBlock = new SystemMemoryBlock<int>(10);
        memoryBlock.CopyFrom(data);

        var span = memoryBlock.AsSpan(2, 5);

        span.Length.Should().Be(5);

        for (var i = 0; i < span.Length; i++)
        {
            span[i].Should().Be(data[i + 2]);
        }
    }

    [Fact]
    public void ThrowException_WhenDisposed()
    {
        var memoryBlock = new SystemMemoryBlock<int>(10);
        memoryBlock.Dispose();

        Action act = () => memoryBlock.AsSpan();

        act.Should().Throw<ObjectDisposedException>();
    }

    [Fact]
    public void ThrowException_WhenUsingLongRange()
    {
        var memoryBlock = new SystemMemoryBlock<int>((long)int.MaxValue + 10);

        Action act = () => memoryBlock.AsSpan();

        act.Should().Throw<InvalidOperationException>();
    }
}