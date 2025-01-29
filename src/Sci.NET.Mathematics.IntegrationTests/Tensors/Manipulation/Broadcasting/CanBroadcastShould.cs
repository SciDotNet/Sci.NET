// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Manipulation.Broadcasting;

public class CanBroadcastShould
{
    public static readonly IEnumerable<object[]> MemberData = GetMemberData();

    private static IEnumerable<object[]> GetMemberData()
    {
        yield return new object[] { Shape.Scalar(), Shape.Scalar(), true };
        yield return new object[] { Shape.Scalar(), Shape.Vector(5), true };
        yield return new object[] { Shape.Scalar(), Shape.Matrix(5, 5), true };
        yield return new object[] { Shape.Scalar(), Shape.Tensor(5, 5, 5), true };

        yield return new object[] { Shape.Vector(5), Shape.Scalar(), false };
        yield return new object[] { Shape.Vector(1), Shape.Vector(5), true };
        yield return new object[] { Shape.Vector(5), Shape.Vector(5), true };
        yield return new object[] { Shape.Vector(5), Shape.Matrix(5, 5), true };
        yield return new object[] { Shape.Vector(4), Shape.Matrix(5, 5), false };
        yield return new object[] { Shape.Vector(5), Shape.Tensor(5, 5, 5), true };
        yield return new object[] { Shape.Vector(4), Shape.Tensor(5, 5, 5), false };

        yield return new object[] { Shape.Matrix(5, 5), Shape.Scalar(), false };
        yield return new object[] { Shape.Matrix(5, 5), Shape.Vector(5), false };
        yield return new object[] { Shape.Matrix(5, 5), Shape.Matrix(5, 5), true };
        yield return new object[] { Shape.Matrix(4, 5), Shape.Matrix(5, 5), false };
        yield return new object[] { Shape.Matrix(5, 5), Shape.Tensor(5, 5, 5), true };
        yield return new object[] { Shape.Matrix(5, 5), Shape.Tensor(5, 6, 5), false };
        yield return new object[] { Shape.Matrix(6, 5), Shape.Tensor(5, 5, 5), false };

        yield return new object[] { Shape.Tensor(5, 5, 5), Shape.Scalar(), false };
        yield return new object[] { Shape.Tensor(5, 5, 5), Shape.Vector(5), false };
        yield return new object[] { Shape.Tensor(5, 5, 5), Shape.Matrix(5, 5), false };
        yield return new object[] { Shape.Tensor(5, 5, 5), Shape.Tensor(5, 5, 5), true };
        yield return new object[] { Shape.Tensor(5, 6, 5), Shape.Tensor(5, 5, 5), false };
        yield return new object[] { Shape.Tensor(5, 5, 5), Shape.Tensor(6, 5, 5, 5), true };
        yield return new object[] { Shape.Tensor(5, 5, 5), Shape.Tensor(5, 6, 5, 5), false };
    }

    [Theory]
    [MemberData(nameof(MemberData))]
    public void ReturnExpectedResult_GivenTwoShapes(Shape original, Shape target, bool expected)
    {
        // Arrange
        var tensor = Tensor.Ones<int>(original);

        // Act
        var actual = tensor.CanBroadcastTo(target);

        // Assert
        actual.Should().Be(expected);
    }
}