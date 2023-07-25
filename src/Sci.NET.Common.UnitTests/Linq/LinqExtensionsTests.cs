// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Linq;

namespace Sci.NET.Common.UnitTests.Linq;

public class LinqExtensionsTests
{
    [Fact]
    public void Product_GivenArray_ReturnsExpectedResult()
    {
        // Arrange
        var array = new int[] { 1, 2, 3, 4, 5 };
        const int expected = 120;

        // Act
        var result = array.Product();

        // Assert
        result.Should().Be(expected);
    }

    [Fact]
    public void Product_GivenEnumerable_ReturnsExpected()
    {
        // Arrange
        var enumerable = Enumerable.Range(1, 5);
        const int expected = 120;

        // Act
        var result = enumerable.Product();

        // Assert
        result.Should().Be(expected);
    }

    [Fact]
    public void Product_GivenEmptyCollection_ReturnsOne()
    {
        // Arrange
        var enumerable = Enumerable.Empty<int>();
        const int expected = 1;

        // Act
        var result = enumerable.Product();

        // Assert
        result.Should().Be(expected);
    }
}