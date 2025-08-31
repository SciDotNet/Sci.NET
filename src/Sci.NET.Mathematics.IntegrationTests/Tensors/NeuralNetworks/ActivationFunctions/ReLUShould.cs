// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.NeuralNetworks.ActivationFunctions;

public class ReLUShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenHalf(IDevice device)
    {
        // ReLU(0.5) = 0.5
        ScalarReLUTest<float>(0.5f, device).Should().BeApproximately(0.5f, 1e-6f);
        ScalarReLUTest<double>(0.5, device).Should().BeApproximately(0.5, 1e-6);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenZero(IDevice device)
    {
        // ReLU(0.0) = 0.0
        ScalarReLUTest<float>(0.0f, device).Should().BeApproximately(0.0f, 1e-6f);
        ScalarReLUTest<double>(0.0, device).Should().BeApproximately(0.0, 1e-6);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenNegativeHalf(IDevice device)
    {
        // ReLU(-0.5) = 0.0
        ScalarReLUTest<float>(-0.5f, device).Should().BeApproximately(0.0f, 1e-6f);
        ScalarReLUTest<double>(-0.5, device).Should().BeApproximately(0.0, 1e-6);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenLargeOnesVector(IDevice device)
    {
        var inputArray = Enumerable.Range(0, 8204).Select(i => (float)(i - 128)).ToArray();
        var expectedArray = inputArray.Select(i => i < 0 ? 0f : i).ToArray();
        using var vector = Tensor.FromArray<float>(inputArray);

        vector.To(device);

        var result = vector.ReLU();

        var resultArray = result.Memory.ToArray();
        for (var i = 0; i < resultArray.Length; i++)
        {
            resultArray[i].Should().BeApproximately(expectedArray[i], 1e-6f);
        }

        result.Should().HaveEquivalentElements(expectedArray);
    }

    [Fact]
    public void Stress_Test()
    {
        var tensor = Tensor.Random.Uniform<float>(new Shape(100, 200), -1f, 1f, seed: 123456).ToTensor();
        var result = Tensor.Zeros<float>(new Shape(100, 200));

        for (var i = 0; i < 100_000; i++)
        {
            tensor.Backend.ActivationFunctions.ReLU(tensor, result);
        }

        result.Should().HaveShape(100, 200);
    }

    private static TNumber ScalarReLUTest<TNumber>(TNumber value, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensor = new Scalar<TNumber>(value);
        tensor.To(device);

        return tensor.ReLU().ToScalar().Value;
    }
}