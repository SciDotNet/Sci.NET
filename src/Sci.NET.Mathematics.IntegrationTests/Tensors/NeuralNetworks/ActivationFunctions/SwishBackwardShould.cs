// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class SwishBackwardShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenFloat(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<float>(new float[] { -2, -1, 0, 1, 2, 3, 4, 5 });
        value.To(device);

        // Act
        var result = value.SwishBackward();

        // Assert
        result
            .Should()
            .HaveShape(8)
            .And
            .HaveApproximatelyEquivalentElements(new float[] { -0.090784244F, 0.07232949F, 0.5F, 0.9276706F, 1.0907842F, 1.0881042F, 1.0526646F, 1.0265474F }, 1e-6f);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenDouble(IDevice device)
    {
        // Arrange
        var value = Tensor.FromArray<double>(new double[] { -2, -1, 0, 1, 2, 3, 4, 5 });
        value.To(device);

        // Act
        var result = value.SwishBackward();

        // Assert
        result
            .Should()
            .HaveShape(8)
            .And
            .HaveApproximatelyEquivalentElements(new double[] { -0.090784244F, 0.07232949F, 0.5F, 0.9276706F, 1.0907842F, 1.0881042F, 1.0526646F, 1.0265474F }, 1e-6);
    }
}