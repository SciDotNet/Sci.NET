// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.LinearAlgebra;

public class MatrixMultiplyShould : IntegrationTestBase
{
    private static Array MatrixMatrixTest<TNumber>(TNumber[,] left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = Tensor.FromArray<TNumber>(left).ToMatrix();
        var rightVector = Tensor.FromArray<TNumber>(right).ToMatrix();
        leftScalar.To(device);
        rightVector.To(device);

        var result = leftScalar.MatrixMultiply(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    private static void MatrixMultiplyTestWithGrad<TNumber>(string safetensorsName, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        // Arrange
        var loadDirectory = $@"{Path.GetDirectoryName(typeof(MatrixMultiplyShould).Assembly.Location)}\Tensors\LinearAlgebra\Examples";
        var tensors = Tensor.LoadSafeTensors<TNumber>($"{loadDirectory}\\{safetensorsName}.safetensors");
        var left = tensors["left"].ToMatrix(requiresGradient: true);
        var right = tensors["right"].ToMatrix(requiresGradient: true);
        var expectedResult = tensors["result"].ToMatrix(requiresGradient: true);
        var expectedLeftGradient = tensors["left_grad"];
        var expectedRightGradient = tensors["right_grad"];
        using var resultGradient = Tensor.Ones<TNumber>(expectedResult.Shape);

        left.To(device);
        right.To(device);
        expectedResult.To(device);
        resultGradient.To(device);
        expectedResult.To(device);
        expectedLeftGradient.To(device);
        expectedRightGradient.To(device);

        // Act
        var result = left.MatrixMultiply(right);
        result.Backward();

        // Assert
        result.Should().HaveApproximatelyEquivalentElements(expectedResult.ToArray(), TNumber.CreateChecked(1e-7f));
        result.Gradient!.Should().NotBeNull();
        result.Gradient!.Should().HaveApproximatelyEquivalentElements(resultGradient.ToArray(), TNumber.CreateChecked(1e-7f));
        left.Gradient!.Should().NotBeNull();
        left.Gradient!.Should().HaveApproximatelyEquivalentElements(expectedLeftGradient.ToArray(), TNumber.CreateChecked(1e-7f));
        right.Gradient!.Should().NotBeNull();
        right.Gradient!.Should().HaveApproximatelyEquivalentElements(expectedRightGradient.ToArray(), TNumber.CreateChecked(1e-7f));
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndMatrix(IDevice device)
    {
        MatrixMatrixTest<float>(
                new float[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new float[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new float[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<double>(
                new double[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new double[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new double[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<byte>(
                new byte[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new byte[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new byte[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<sbyte>(
                new sbyte[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new sbyte[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new sbyte[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<ushort>(
                new ushort[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new ushort[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new ushort[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<short>(
                new short[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new short[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new short[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<uint>(
                new uint[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new uint[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new uint[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<int>(
                new int[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new int[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new int[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<ulong>(
                new ulong[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new ulong[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new ulong[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<long>(
                new long[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new long[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new long[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<BFloat16>(
                new BFloat16[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new BFloat16[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new BFloat16[,] { { 30, 30 }, { 30, 30 } });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenExample1(IDevice device)
    {
        using var leftTensor = Tensor
            .FromArray<int>(new int[,] { { 0, 4, 8, 12, 16, 20, 24, 28, 32, 36 }, { 1, 5, 9, 13, 17, 21, 25, 29, 33, 37 }, { 2, 6, 10, 14, 18, 22, 26, 30, 34, 38 }, { 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 } })
            .ToMatrix();
        using var rightTensor = Tensor
            .FromArray<int>(new int[,] { { 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 }, { 8, 9 }, { 10, 11 }, { 12, 13 }, { 14, 15 }, { 16, 17 }, { 18, 19 } })
            .ToMatrix();

        leftTensor.To(device);
        rightTensor.To(device);

        var result = leftTensor.MatrixMultiply(rightTensor);

        result.Should().HaveEquivalentElements(new int[,] { { 2280, 2460 }, { 2370, 2560 }, { 2460, 2660 }, { 2550, 2760 } });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenPyTorchExample1(IDevice device)
    {
        MatrixMultiplyTestWithGrad<float>("MatrixMultiply_1", device);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenPyTorchExample2(IDevice device)
    {
        MatrixMultiplyTestWithGrad<float>("MatrixMultiply_2", device);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenPyTorchExample3(IDevice device)
    {
        MatrixMultiplyTestWithGrad<float>("MatrixMultiply_3", device);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenPyTorchExample4(IDevice device)
    {
        MatrixMultiplyTestWithGrad<float>("MatrixMultiply_4", device);
    }
}