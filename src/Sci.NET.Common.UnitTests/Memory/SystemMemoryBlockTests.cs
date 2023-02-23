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

    [Fact]
    public void Getter_GivenIndexOutOfRange_ShouldThrow()
    {
        var source = new SystemMemoryBlock<float>(10);

        var act = () => _ = source[10];

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Theory]
    [InlineData(10)]
    [InlineData(100)]
    [InlineData(-1)]
    public void Setter_GivenIndexOutOfRange_ShouldThrow(int index)
    {
        var source = new SystemMemoryBlock<float>(10);

        var act = () => source[index] = 0;

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Equals_WhenCalledWithReferenceEquality_ShouldReturnTrue()
    {
#pragma warning disable RCS1124
        var left = new SystemMemoryBlock<float>(10);
        var right = left;
#pragma warning restore RCS1124

        left.Equals(right).Should().BeTrue();
        (left == right).Should().BeTrue();
        (left != right).Should().BeFalse();
    }

    [Fact]
    public void Equals_WhenCalledWithReferenceInequality_ShouldReturnFalse()
    {
        var left = new SystemMemoryBlock<float>(10);
        var right = new SystemMemoryBlock<float>(10);

        left.Equals(right).Should().BeFalse();
        (left == right).Should().BeFalse();
        (left != right).Should().BeTrue();
    }

    [Fact]
    public void Equals_WhenCalledWithNull_ShouldReturnFalse()
    {
        var left = new SystemMemoryBlock<float>(10);

#pragma warning disable CA1508
        left.Equals((object?)null).Should().BeFalse();
#pragma warning restore CA1508
    }

    [Fact]
    public void Equals_WhenCalledWithDifferentType_ShouldReturnFalse()
    {
        var left = new SystemMemoryBlock<float>(10);

        left.Equals(new object()).Should().BeFalse();
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
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
        })]
    public void ToArray_WhenCalled_ShouldReturnSameData(float[] data)
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
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
        })]
    public void CopyFrom_WhenCalled_ShouldCopyData(float[] data)
    {
        var destination = new SystemMemoryBlock<float>(data.LongLength);

        destination.CopyFrom(data);

        destination.ToArray().Should().BeEquivalentTo(data);
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
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
        })]
    public void CopyTo_WhenCalled_ShouldCopyData(float[] data)
    {
        var source = new SystemMemoryBlock<float>(data);
        var destination = new SystemMemoryBlock<float>(data.LongLength);

        source.CopyTo(destination);

        destination.ToArray().Should().BeEquivalentTo(data);
    }
}