// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory.SystemMemoryBlock;

public class ConstructorShould
{
    [Fact]
    public unsafe void AllocateMemory_GivenDataLength()
    {
        // Arrange & Act
        var block = new SystemMemoryBlock<int>(10);

        // Assert
        block.Length.Should().Be(10);
        block.IsDisposed.Should().BeFalse();
        ((nint)block.ToPointer()).Should().BeGreaterThan(0);

        foreach (var element in block)
        {
            element.Should().Be(0);
        }
    }

    [Fact]
    public unsafe void AllocateMemory_WhenCalledWithZeroLength()
    {
        // Arrange & Act
        var block = new SystemMemoryBlock<int>(0);

        // Assert
        block.Length.Should().Be(0);
        block.IsDisposed.Should().BeFalse();
        ((nint)block.ToPointer()).Should().BeGreaterThan(0);
    }

    [Fact]
    public unsafe void AllocateAndCopyMemory_GivenData()
    {
        // Arrange
        var data = new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        // Act
        var block = new SystemMemoryBlock<int>(data);

        // Assert
        block.Length.Should().Be(10);
        block.IsDisposed.Should().BeFalse();
        ((nint)block.ToPointer()).Should().BeGreaterThan(0);

        var index = 0;

        foreach (var element in block)
        {
            element.Should().Be(data[index++]);
        }
    }
}