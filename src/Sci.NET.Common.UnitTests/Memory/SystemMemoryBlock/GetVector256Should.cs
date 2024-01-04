// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.Intrinsics;
using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory.SystemMemoryBlock;

public class GetVector256Should
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    public void GetCorrectly_GivenIntWithOffset(int offset)
    {
        var data = new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        var block = new SystemMemoryBlock<int>(data.Length);

        block.CopyFrom(data);

        var result = block.GetVector256(offset);

        for (var i = 0; i < Vector256<int>.Count; i++)
        {
            result[i].Should().Be(offset + i + 1);
        }
    }

    [Fact]
    public void Throw_WhenLengthIsLessThanVector256()
    {
        var block = new SystemMemoryBlock<int>(6);

        var act = () => block.GetVector256(0);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Throw_WhenOffsetIsLessThanZero()
    {
        var block = new SystemMemoryBlock<int>(6);

        var act = () => block.GetVector256(-1);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Throw_WhenOffsetIsGreaterThanLength()
    {
        var block = new SystemMemoryBlock<int>(6);

        var act = () => block.GetVector256(7);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}