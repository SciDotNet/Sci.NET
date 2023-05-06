// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sut = Sci.NET.Mathematics.Tensors.Shape;

namespace Sci.NET.Mathematics.UnitTests.Tensors.Shape;

public class EqualityOperatorsShould
{
    [Fact]
    public void ReturnTrue_GivenEqualShapes()
    {
        // Arrange
        var shape1 = new Sut(3, 4, 5);
        var shape2 = new Sut(3, 4, 5);

        // Act & Assert
        (shape1 == shape2).Should().BeTrue();
        (shape1 != shape2).Should().BeFalse();
    }

    [Fact]
    public void ReturnFalse_GivenUnequalShapes()
    {
        var shape1 = new Sut(3, 4, 5);
        var shape2 = new Sut(3, 4, 6);

        (shape1 == shape2).Should().BeFalse();
        (shape1 != shape2).Should().BeTrue();
    }
}