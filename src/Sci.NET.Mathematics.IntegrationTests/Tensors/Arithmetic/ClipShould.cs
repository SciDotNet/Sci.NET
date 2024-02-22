// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Arithmetic;

public class ClipShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ClipToGivenValues_GivenScalar(IDevice device)
    {
        // Arrange
        var input = new Scalar<float>(1);
        const float min = 2.5f;
        const float max = 5.5f;

        input.To(device);

        // Act
        var result = input.Clip(min, max);

        // Assert
        result.Should().BeOfType<Scalar<float>>();

        result
            .Should()
            .HaveShape()
            .And
            .HaveApproximatelyEquivalentElements(new float[] { 2.5f }, 1e-6f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ClipToGivenValues_GivenVector(IDevice device)
    {
        // Arrange
        var input = Tensor.FromArray<float>(new float[] { 1, 2, 3, 4, 5, 6 }).ToVector();
        const float min = 2.5f;
        const float max = 5.5f;

        input.To(device);

        // Act
        var result = input.Clip(min, max);

        // Assert
        result.Should().BeOfType<Vector<float>>();

        result
            .Should()
            .HaveShape(6)
            .And
            .HaveApproximatelyEquivalentElements(new float[] { 2.5f, 2.5f, 3.0f, 4.0f, 5.0f, 5.5f }, 1e-6f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ClipToGivenValues_GivenMatrix(IDevice device)
    {
        // Arrange
        var input = Tensor.FromArray<float>(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }).ToMatrix();
        const float min = 2.5f;
        const float max = 5.5f;

        input.To(device);

        // Act
        var result = input.Clip(min, max);

        // Assert
        result.Should().BeOfType<Matrix<float>>();

        result
            .Should()
            .HaveShape(3, 2)
            .And
            .HaveApproximatelyEquivalentElements(new float[,] { { 2.5f, 2.5f }, { 3.0f, 4.0f }, { 5.0f, 5.5f } }, 1e-6f);
    }
}