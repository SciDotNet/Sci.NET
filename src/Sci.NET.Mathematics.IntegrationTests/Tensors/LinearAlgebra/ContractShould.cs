// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.LinearAlgebra;

public class ContractShould : IntegrationTestBase
{
    private static void PyTorchTest<TNumber>(string safetensorsName, IDevice device, int[] leftIndices, int[] rightIndices)
        where TNumber : unmanaged, INumber<TNumber>
    {
        // Arrange
        var loadDirectory = $@"{Path.GetDirectoryName(typeof(MatrixMultiplyShould).Assembly.Location)}\Tensors\LinearAlgebra\Examples";
        var tensors = Tensor.LoadSafeTensors<TNumber>($"{loadDirectory}\\{safetensorsName}.safetensors");
        var left = tensors["left"].RecreateWithGradient();
        var right = tensors["right"];
        var expectedResult = tensors["result"];

        left.To(device);
        right.To(device);
        expectedResult.To(device);
        expectedResult.To(device);

        // Act
        var result = left.Contract(right, leftIndices, rightIndices);
        result.Backward();

        // Assert
        result.Should().HaveApproximatelyEquivalentElements(expectedResult.ToArray(), TNumber.CreateChecked(1e-7f));
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_GivenTwoTensorsWithSameShape(IDevice device)
    {
        var left = Tensor.FromArray<float>(new float[,,] { { { 0, 1 }, { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 }, { 10, 11 } }, { { 12, 13 }, { 14, 15 }, { 16, 17 } }, { { 18, 19 }, { 20, 21 }, { 22, 23 } } });
        var right = Tensor.FromArray<float>(new float[,,] { { { 0, 1 }, { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 }, { 10, 11 } }, { { 12, 13 }, { 14, 15 }, { 16, 17 } }, { { 18, 19 }, { 20, 21 }, { 22, 23 } } });

        left.To(device);
        right.To(device);

        var result = left.Contract(right, new int[] { 1, 2 }, new int[] { 1, 2 });

        result
            .Should()
            .HaveShape(4, 4)
            .And
            .HaveEquivalentElements(new float[,] { { 55, 145, 235, 325 }, { 145, 451, 757, 1063 }, { 235, 757, 1279, 1801 }, { 325, 1063, 1801, 2539 } });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_GivenPyTorchExample1(IDevice device)
    {
        PyTorchTest<long>("Contract_[[1]_[0]]_1", device, new int[] { 1 }, new int[] { 0 });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_GivenPyTorchExample2(IDevice device)
    {
        PyTorchTest<long>("Contract_[[1]_[0]]_2", device, new int[] { 1 }, new int[] { 0 });
    }
}