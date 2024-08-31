// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Manipulation.Casting;

public class CastShould
{
    [Fact]
    public void ReturnExpectedResult_GivenScalar()
    {
        // Arrange
        var tensor = Tensor.Ones<int>(Shape.Scalar()).ToScalar();

        // Act
        var actual = tensor.Cast<int, float>();

        // Assert
        actual.Value.Should().Be(1.0f);
    }

    [Fact]
    public void ReturnExpectedResult_GivenVector()
    {
        // Arrange
        var tensor = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5 }).ToVector();

        // Act
        var actual = tensor.Cast<int, float>();

        // Assert
        actual.Should().HaveEquivalentElements(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
    }

    [Fact]
    public void ReturnExpectedResult_GivenMatrix()
    {
        // Arrange
        var tensor = Tensor.FromArray<int>(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }).ToMatrix();

        // Act
        var actual = tensor.Cast<int, float>();

        // Assert
        actual.Should().HaveEquivalentElements(new float[,] { { 1.0f, 2.0f }, { 3.0f, 4.0f }, { 5.0f, 6.0f } });
    }

    [Fact]
    public void ReturnExpectedResult_GivenTensor()
    {
        // Arrange
        var tensor = Tensor.FromArray<int>(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } });

        // Act
        var actual = tensor.Cast<int, float>();

        // Assert
        actual.Should().HaveEquivalentElements(new float[,,] { { { 1.0f, 2.0f }, { 3.0f, 4.0f } }, { { 5.0f, 6.0f }, { 7.0f, 8.0f } } });
    }

    [Fact]
    public void ReturnExpectedResult_GivenLargeITensor()
    {
        var tensor = Tensor.FromArray<int>(Enumerable.Range(0, 24 * 25 * 26).ToArray()).Reshape(24, 25, 26);
        var expected = Tensor.FromArray<float>(Enumerable.Range(0, 24 * 25 * 26).Select(x => (float)x).ToArray()).Reshape(24, 25, 26);

        // Act
        var actual = tensor.Cast<int, float>();

        // Assert
        actual.Should().HaveEquivalentElements(expected.ToArray());
    }
}