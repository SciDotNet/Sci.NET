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
        var left = tensors["left"].WithGradient();
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
        var left = Tensor.FromArray<float>(Enumerable.Range(0, 4 * 3 * 2).Select(x => (float)x).ToArray()).Reshape(4, 3, 2);
        var right = Tensor.FromArray<float>(Enumerable.Range(0, 4 * 3 * 2).Select(x => (float)x).ToArray()).Reshape(4, 3, 2);

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
    public void ReturnsCorrectResult_Example1(IDevice device)
    {
        // Arrange
        var left = Tensor.FromArray<int>(Enumerable.Range(0, 2 * 3 * 4).ToArray()).Reshape(2, 3, 4);
        var right = Tensor.FromArray<int>(Enumerable.Range(0, 3 * 4 * 5).ToArray()).Reshape(3, 4, 5);

        var expected = new int[,,,]
        {
            {
                { { 400, 412, 424, 436, 448 }, { 460, 472, 484, 496, 508 }, { 520, 532, 544, 556, 568 }, { 580, 592, 604, 616, 628 } },
                { { 460, 475, 490, 505, 520 }, { 535, 550, 565, 580, 595 }, { 610, 625, 640, 655, 670 }, { 685, 700, 715, 730, 745 } },
                { { 520, 538, 556, 574, 592 }, { 610, 628, 646, 664, 682 }, { 700, 718, 736, 754, 772 }, { 790, 808, 826, 844, 862 } },
                { { 580, 601, 622, 643, 664 }, { 685, 706, 727, 748, 769 }, { 790, 811, 832, 853, 874 }, { 895, 916, 937, 958, 979 } }
            },
            {
                { { 1120, 1168, 1216, 1264, 1312 }, { 1360, 1408, 1456, 1504, 1552 }, { 1600, 1648, 1696, 1744, 1792 }, { 1840, 1888, 1936, 1984, 2032 } },
                { { 1180, 1231, 1282, 1333, 1384 }, { 1435, 1486, 1537, 1588, 1639 }, { 1690, 1741, 1792, 1843, 1894 }, { 1945, 1996, 2047, 2098, 2149 } },
                { { 1240, 1294, 1348, 1402, 1456 }, { 1510, 1564, 1618, 1672, 1726 }, { 1780, 1834, 1888, 1942, 1996 }, { 2050, 2104, 2158, 2212, 2266 } },
                { { 1300, 1357, 1414, 1471, 1528 }, { 1585, 1642, 1699, 1756, 1813 }, { 1870, 1927, 1984, 2041, 2098 }, { 2155, 2212, 2269, 2326, 2383 } }
            }
        };

        left.To(device);
        right.To(device);

        // Act
        var result = left.Contract(right, new int[] { 1 }, new int[] { 0 });

        // Assert
        result.Should().HaveEquivalentElements(expected);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_Example2(IDevice device)
    {
        // Arrange
        var left = Tensor.FromArray<int>(Enumerable.Range(0, 2 * 5 * 4).ToArray()).Reshape(2, 5, 4);
        var right = Tensor.FromArray<int>(Enumerable.Range(0, 2 * 5 * 2).ToArray()).Reshape(2, 5, 2);

        var expected = new int[,] { { 2280, 2460 }, { 2370, 2560 }, { 2460, 2660 }, { 2550, 2760 } };

        left.To(device);
        right.To(device);

        // Act
        var result = left.Contract(right, new int[] { 0, 1 }, new int[] { 0, 1 });

        // Assert
        result.Should().HaveEquivalentElements(expected);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_GivenPyTorchExample1(IDevice device)
    {
        PyTorchTest<int>("Contract_[[1]_[0]]_1", device, new int[] { 1 }, new int[] { 0 });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnsCorrectResult_GivenPyTorchExample2(IDevice device)
    {
        PyTorchTest<int>("Contract_[[1]_[0]]_2", device, new int[] { 1 }, new int[] { 0 });
    }
}