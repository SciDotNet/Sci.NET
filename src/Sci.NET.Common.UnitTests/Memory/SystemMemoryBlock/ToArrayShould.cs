// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory.SystemMemoryBlock;

public class ToArrayShould
{
    [Theory]
    [InlineData(new int[] { })]
    [InlineData(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 })]
    [InlineData(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 })]
    [InlineData(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 })]
    public void ToArray_WhenCalled_ShouldCopyData(int[] data)
    {
        // Arrange
        var source = new SystemMemoryBlock<int>(data);

        // Act
        var result = source.ToArray();

        // Assert
        result.Should().BeEquivalentTo(data);
    }
}