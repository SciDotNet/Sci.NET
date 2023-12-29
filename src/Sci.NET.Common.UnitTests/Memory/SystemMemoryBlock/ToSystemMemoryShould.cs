// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory.SystemMemoryBlock;

public class ToSystemMemoryShould
{
    [Fact]
    public void ReturnSameInstance()
    {
        // Arrange
        var block = new SystemMemoryBlock<int>(10);

        // Act
        var result = block.ToSystemMemory();

        // Assert
        result.Should().BeSameAs(block);
    }
}