// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Runtime.Intrinsics;
using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory.SystemMemoryBlock;

public class SetVector256Should
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
    public void SetResultCorrectly_GivenIndex(int offset)
    {
        var data = new int[16];
        var block = new SystemMemoryBlock<int>(data.Length);

        var result = Vector256.Create(
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8);

        block.SetVector256(offset, result);

        for (var i = 0; i < 8; i++)
        {
            block[offset + i].Should().Be(i + 1);
        }

        for (var i = 0; i < offset; i++)
        {
            block[i].Should().Be(0);
        }

        for (var i = offset + 8; i < data.Length; i++)
        {
            block[i].Should().Be(0);
        }
    }

    [Fact]
    public void Throw_WhenLengthIsLessThanVector256()
    {
        var block = new SystemMemoryBlock<int>(6);

        var act = () => block.SetVector256(0, Vector256<int>.Zero);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Throw_WhenIndexIsNegative()
    {
        var block = new SystemMemoryBlock<int>(16);

        var act = () => block.SetVector256(-1, Vector256<int>.Zero);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Throw_WhenIndexIsGreaterThanLength()
    {
        var block = new SystemMemoryBlock<int>(16);

        var act = () => block.SetVector256(16, Vector256<int>.Zero);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}