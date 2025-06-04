// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class ReLUBackwardShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenHalf(IDevice device)
    {
        // ReLU'(0.5) = 1.0
        ScalarReLUBackwardTest<float>(0.5f, device).Should().BeApproximately(1.0f, 1e-6f);
        ScalarReLUBackwardTest<double>(0.5, device).Should().BeApproximately(1.0, 1e-6);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenZero(IDevice device)
    {
        // ReLU'(0.0) = 0.0
        ScalarReLUBackwardTest<float>(0.0f, device).Should().BeApproximately(0.0f, 1e-6f);
        ScalarReLUBackwardTest<double>(0.0, device).Should().BeApproximately(0.0, 1e-6);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenNegativeHalf(IDevice device)
    {
        // ReLU'(-0.5) = 0.0
        ScalarReLUBackwardTest<float>(-0.5f, device).Should().BeApproximately(0.0f, 1e-6f);
        ScalarReLUBackwardTest<double>(-0.5, device).Should().BeApproximately(0.0, 1e-6);
    }

    private static TNumber ScalarReLUBackwardTest<TNumber>(TNumber value, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensor = new Scalar<TNumber>(value);
        tensor.To(device);

        return tensor.ReLUBackward().ToScalar().Value;
    }
}