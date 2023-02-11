// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory;

public class SystemMemoryBlockTests
{
    [Fact]
    public unsafe void Ctor_WhenCalled_ShouldAllocateMemory()
    {
        const int count = 10;
        var handle = new SystemMemoryBlock<int>(count);

        ((nuint)handle.ToPointer()).Should().BeGreaterThan(0);
        handle.Length.Should().Be(count);
    }

    [Theory]
    [InlineData(
        new float[]
        {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        })]
    [InlineData(
        new float[]
        {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        })]
    [InlineData(
        new float[]
        {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32
        })]
    public void ToArray_WhenCalled_ShouldCopyData(float[] data)
    {
        var source = new SystemMemoryBlock<float>(data);

        source.ToArray().Should().BeEquivalentTo(data);
    }

    [Theory]
    [InlineData(
        new float[]
        {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        })]
    [InlineData(
        new float[]
        {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        })]
    [InlineData(
        new float[]
        {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32
        })]
    public void Setter_WhenCalled_ShouldSetData(float[] data)
    {
        using var source = new SystemMemoryBlock<float>(data.LongLength);

        for (var i = 0; i < data.LongLength; i++)
        {
            source[i] = data[i];
        }

        source.ToArray().Should().BeEquivalentTo(data);
    }
}