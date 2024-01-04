// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Sci.NET.Common.Memory;

namespace Sci.NET.Common.UnitTests.Memory.SystemMemoryBlock;

#pragma warning disable CS1718

[SuppressMessage(
    "Microsoft.CodeAnalysis.CSharp",
    "CS1718:Comparison made to same variable; did you mean to compare something else?",
    Justification = "Test")]
public class EqualityMembersShould
{
    [Fact]
    [SuppressMessage("ReSharper", "EqualExpressionComparison", Justification = "Test")]
    public void ReturnTrue_GivenSameReference()
    {
        // Arrange
        using var block = new SystemMemoryBlock<int>(10);

        // Act
        var methodResult = block.Equals(block);
        var operatorResult = block == block;
        var negatedOperatorResult = block != block;

        // Assert
        methodResult.Should().BeTrue();
        operatorResult.Should().BeTrue();
        negatedOperatorResult.Should().BeFalse();
    }

    [Fact]
    public void ReturnFalse_GivenDifferentReference()
    {
        // Arrange
        var block1 = new SystemMemoryBlock<int>(10);
        var block2 = new SystemMemoryBlock<int>(10);

        // Act
        var methodResult = block1.Equals(block2);
        var operatorResult = block1 == block2;
        var negatedOperatorResult = block1 != block2;

        // Assert
        methodResult.Should().BeFalse();
        operatorResult.Should().BeFalse();
        negatedOperatorResult.Should().BeTrue();
    }

    [Fact]
    public void ReturnFalse_GivenNull()
    {
        // Arrange
        var block = new SystemMemoryBlock<int>(10);

        // Act
#pragma warning disable CA1508
        var result = block.Equals(null);
#pragma warning restore CA1508

        // Assert
        result.Should().BeFalse();
    }

    [Fact]
    public void ReturnFalse_GivenDifferentType()
    {
        // Arrange
        var block = new SystemMemoryBlock<int>(10);

        // Act
        // ReSharper disable once SuspiciousTypeConversion.Global - Test
        var result = block.Equals(10);

        // Assert
        result.Should().BeFalse();
    }

    [Fact]
    public void ThrowsException_WhenLeftDisposed()
    {
        // Arrange
        var block1 = new SystemMemoryBlock<int>(10);
        var block2 = new SystemMemoryBlock<int>(10);

        block1.Dispose();

        // Act
        var act = () => block1.Equals(block2);

        // Assert
        act.Should().Throw<ObjectDisposedException>();
    }

    [Fact]
    public void ReturnFalse_WhenRightDisposed()
    {
        // Arrange
        var block1 = new SystemMemoryBlock<int>(10);
        var block2 = new SystemMemoryBlock<int>(10);

        block2.Dispose();

        // Act
        var act = () => block1.Equals(block2);

        // Assert
        act.Should().Throw<ObjectDisposedException>();
    }
}