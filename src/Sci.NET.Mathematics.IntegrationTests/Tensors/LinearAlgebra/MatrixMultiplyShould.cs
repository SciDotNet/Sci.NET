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
        var leftTensor = Tensor.FromArray<TNumber>(left).WithGradient().ToMatrix();
        var rightTensor = Tensor.FromArray<TNumber>(right).WithGradient().ToMatrix();
        leftTensor.To(device);
        rightTensor.To(device);

        var result = leftTensor.MatrixMultiply(rightTensor);

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
        result.Should().HaveApproximatelyEquivalentElements(expectedResult.ToArray(), TNumber.CreateChecked(1e-4f));
        result.Gradient!.Should().NotBeNull();
        result.Gradient!.Should().HaveApproximatelyEquivalentElements(resultGradient.ToArray(), TNumber.CreateChecked(1e-4f));
        left.Gradient!.Should().NotBeNull();
        left.Gradient!.Should().HaveApproximatelyEquivalentElements(expectedLeftGradient.ToArray(), TNumber.CreateChecked(1e-4f));
        right.Gradient!.Should().NotBeNull();
        right.Gradient!.Should().HaveApproximatelyEquivalentElements(expectedRightGradient.ToArray(), TNumber.CreateChecked(1e-4f));
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenFloatMatrixAndMatrix(IDevice device)
    {
        MatrixMatrixTest<float>(
                new float[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new float[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new float[,] { { 30, 30 }, { 30, 30 } });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenDoubleMatrixAndMatrix(IDevice device)
    {
        MatrixMatrixTest<double>(
                new double[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new double[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new double[,] { { 30, 30 }, { 30, 30 } });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenByteMatrixAndMatrix(IDevice device)
    {
        MatrixMatrixTest<byte>(
                new byte[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new byte[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new byte[,] { { 30, 30 }, { 30, 30 } });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenSByteMatrixAndMatrix(IDevice device)
    {
        MatrixMatrixTest<sbyte>(
                new sbyte[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new sbyte[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new sbyte[,] { { 30, 30 }, { 30, 30 } });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenUShortMatrixAndMatrix(IDevice device)
    {
        MatrixMatrixTest<ushort>(
                new ushort[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new ushort[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new ushort[,] { { 30, 30 }, { 30, 30 } });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenShortMatrixAndMatrix(IDevice device)
    {
        MatrixMatrixTest<short>(
                new short[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new short[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new short[,] { { 30, 30 }, { 30, 30 } });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenUIntMatrixAndMatrix(IDevice device)
    {
        MatrixMatrixTest<uint>(
                new uint[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new uint[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new uint[,] { { 30, 30 }, { 30, 30 } });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenIntMatrixAndMatrix(IDevice device)
    {
        MatrixMatrixTest<int>(
                new int[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new int[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new int[,] { { 30, 30 }, { 30, 30 } });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenULongMatrixAndMatrix(IDevice device)
    {
        MatrixMatrixTest<ulong>(
                new ulong[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new ulong[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new ulong[,] { { 30, 30 }, { 30, 30 } });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenLongMatrixAndMatrix(IDevice device)
    {
        MatrixMatrixTest<long>(
                new long[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new long[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new long[,] { { 30, 30 }, { 30, 30 } });
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenBFloat16MatrixAndMatrix(IDevice device)
    {
        MatrixMatrixTest<BFloat16>(
                new BFloat16[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new BFloat16[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new BFloat16[,] { { 30, 30 }, { 30, 30 } });
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
        MatrixMultiplyTestWithGrad<double>("MatrixMultiply_3", device);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenPyTorchExample4(IDevice device)
    {
        MatrixMultiplyTestWithGrad<double>("MatrixMultiply_4", device);
    }
}