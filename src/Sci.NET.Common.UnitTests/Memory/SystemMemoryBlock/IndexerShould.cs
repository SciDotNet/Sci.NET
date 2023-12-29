// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory.SystemMemoryBlock;

public class IndexerShould
{
    [Theory]
    [InlineData(10)]
    [InlineData(100)]
    [InlineData(1000)]
    [InlineData(100000)]
    public void GivenValidIndex_ReturnCorrectValue(int length)
    {
        var array = Enumerable.Range(0, length).Select(x => (byte)x).ToArray();
        var memoryBlock = new SystemMemoryBlock<byte>(length);
        memoryBlock.CopyFrom(array);

        for (var i = 0; i < length; i++)
        {
            array[i].Should().Be(memoryBlock[i]);
        }
    }

    [Theory]
    [InlineData(10)]
    [InlineData(-1)]
    public void GivenInvalidIndex_ThrowIndexOutOfRangeException(int queryIndex)
    {
        var memoryBlock = new SystemMemoryBlock<byte>(10);

        var act = () => _ = memoryBlock[queryIndex];

        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}