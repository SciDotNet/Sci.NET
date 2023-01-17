// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory.ReferenceCounting;

namespace Sci.NET.Common.UnitTests.Memory.ReferenceCounting;

public class ReferenceCountTests
{
    [Fact]
    public void Ctor_WhenCalled_SetsCountToZeroAndCreatesNewId()
    {
        // Arrange & Act
        var referenceCount1 = new ReferenceCount();
        var referenceCount2 = new ReferenceCount();

        // Assert
        referenceCount1.IsZero()
            .Should()
            .BeTrue();

        referenceCount2.IsZero()
            .Should()
            .BeTrue();
    }

    [Fact]
    public void Increment_WhenCalled_IncrementsCount()
    {
        // Arrange
        var referenceCount = new ReferenceCount();

        // Act
        referenceCount.Increment();

        // Assert
        referenceCount.IsZero()
            .Should()
            .BeFalse();
    }

    [Fact]
    public void Increment_WhenDisposed_ThrowsException()
    {
        // Arrange
        var referenceCount = new ReferenceCount();
        referenceCount.Dispose();

        // Act
        var act = () => referenceCount.Increment();

        // Assert
        act.Should()
            .Throw<ObjectDisposedException>();
    }

    [Fact]
    public void Decrement_WhenCalled_DecrementsCount()
    {
        // Arrange
        var referenceCount = new ReferenceCount();

        // Act
        referenceCount.Increment();
        referenceCount.Decrement();

        // Assert
        referenceCount.IsZero()
            .Should()
            .BeTrue();
    }

    [Fact]
    public void Decrement_WhenDisposed_ThrowsObjectDisposedException()
    {
        // Arrange
        var referenceCount = new ReferenceCount();
        referenceCount.Dispose();

        // Act
        var act = () => referenceCount.Decrement();

        // Assert
        act.Should()
            .Throw<ObjectDisposedException>();
    }

    [Theory]
    [InlineData(0, 0, true)]
    [InlineData(0, 1, false)]
    [InlineData(1, 0, false)]
    [InlineData(1, 1, true)]
    public void CountEquals_WhenCalled_ReturnsExpectedResult(int left, int right, bool expected)
    {
        // Arrange
        var referenceCount1 = new ReferenceCount();
        var referenceCount2 = new ReferenceCount();

        // Act
        for (var i = 0; i < left; i++)
        {
            referenceCount1.Increment();
        }

        for (var i = 0; i < right; i++)
        {
            referenceCount2.Increment();
        }

        // Assert
        referenceCount1.CountEquals(referenceCount2)
            .Should()
            .Be(expected);
    }

    [Fact]
    public void CountEquals_WhenCalled_ReturnsTrueWhenCountsAreEqual()
    {
        // Arrange
        var referenceCount1 = new ReferenceCount();
        var referenceCount2 = new ReferenceCount();

        // Act
        referenceCount1.Increment();
        referenceCount2.Increment();

        // Assert
        referenceCount1.CountEquals(referenceCount2)
            .Should()
            .BeTrue();
    }

    [Fact]
    public void GetHashCode_WhenCalled_ReturnsHashCode()
    {
        // Arrange
        var referenceCount = new ReferenceCount();

        // Act & Assert
        referenceCount.GetHashCode()
            .Should()
            .NotBe(0);
    }

    [Fact]
    public void GetHashCode_WhenDisposed_ThrowsException()
    {
        // Arrange
        var referenceCount = new ReferenceCount();
        referenceCount.Dispose();

        // Act
        var act = () => referenceCount.GetHashCode();

        // Assert
        act.Should()
            .Throw<ObjectDisposedException>();
    }

    [Fact]
    public void Dispose_WhenCalled_Disposes()
    {
        // Arrange
        var referenceCount = new ReferenceCount();
        referenceCount.Dispose();

        // Act
        var act = () => referenceCount.IsZero();

        // Assert
        act.Should()
            .Throw<ObjectDisposedException>();
    }

    [Fact]
    public void Equals_WhenCalled_ReturnsFalseWhenDifferentInstances()
    {
        // Arrange
        var referenceCount1 = new ReferenceCount();
        var referenceCount2 = new ReferenceCount();

        // Act
        referenceCount1.Increment();
        referenceCount2.Increment();

        // Assert
        referenceCount1.Equals(referenceCount2)
            .Should()
            .BeFalse();
    }

    [Fact]
    public void Equals_WhenCalled_ReturnsFalseWhenSameInstance()
    {
        // Arrange
        var referenceCount1 = new ReferenceCount();
        var referenceCount2 = referenceCount1;

        // Act
        referenceCount1.Increment();
        referenceCount2.Increment();

        // Assert
        referenceCount1.Equals(referenceCount2)
            .Should()
            .BeTrue();
    }

    [Fact]
    public void Equals_WhenCalled_ThrowsWhenDisposed()
    {
        // Arrange
        var referenceCount1 = new ReferenceCount();
        var referenceCount2 = new ReferenceCount();
        referenceCount1.Dispose();

        // Act
        var act = () => referenceCount1.Equals(referenceCount2);

        // Assert
        act.Should()
            .Throw<ObjectDisposedException>();
    }

    [Fact]
    public void Equals_WhenCalled_ThrowsWhenOtherIsDisposed()
    {
        // Arrange
        var referenceCount1 = new ReferenceCount();
        var referenceCount2 = new ReferenceCount();
        referenceCount2.Dispose();

        // Act
        var act = () => referenceCount1.Equals(referenceCount2);

        // Assert
        act.Should()
            .Throw<ObjectDisposedException>();
    }

    [Fact]
    public void Equals_WhenOtherIsNull_ReturnsFalse()
    {
        // Arrange
        var referenceCount1 = new ReferenceCount();

        // Act & Assert
#pragma warning disable CA1508
        referenceCount1.Equals(null)
#pragma warning restore CA1508
            .Should()
            .BeFalse();
    }
}