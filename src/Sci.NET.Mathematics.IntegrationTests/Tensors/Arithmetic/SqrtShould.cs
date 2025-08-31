// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Arithmetic;

public class SqrtShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenFp32Scalar(IDevice device)
    {
        // Arrange
        var input = new Scalar<float>(4.0f);
        input.To(device);

        // Act
        var result = input.Sqrt();

        // Assert
        result.Value.Should().BeApproximately(2.0f, 1e-6f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenFp64Vector(IDevice device)
    {
        // Arrange
        var input = Tensor.FillWith(9.0f, new Shape(100));
        input.To(device);

        // Act
        var result = input.Sqrt();

        // Assert
        result.Should().HaveShape(100);
        result.Should().HaveEquivalentElements(Enumerable.Repeat(3.0, 100).Select(x => (float)x).ToArray());
    }
}