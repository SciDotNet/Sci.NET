// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class SigmoidBackwardShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalar(IDevice device)
    {
        // Sig'(0.5) = 0.235003712201594498660
        ScalarSigmoidBackwardTest<float>(0.5f, device).Should().BeApproximately(0.235003712201594498660f, 1e-6f);
        ScalarSigmoidBackwardTest<double>(0.5, device).Should().BeApproximately(0.235003712201594498660, 1e-6);

        // Sig'(0.0) = 0.25
        ScalarSigmoidBackwardTest<float>(0.0f, device).Should().BeApproximately(0.25f, 1e-6f);
        ScalarSigmoidBackwardTest<double>(0.0, device).Should().BeApproximately(0.25, 1e-6);
    }

    private static TNumber ScalarSigmoidBackwardTest<TNumber>(TNumber value, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var tensor = new Scalar<TNumber>(value);
        tensor.To(device);

        return tensor.SigmoidBackward().ToScalar().Value;
    }
}