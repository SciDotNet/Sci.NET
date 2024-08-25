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
    private readonly string _safetensorsLoadDirectory;

    public MatrixMultiplyShould()
    {
        _safetensorsLoadDirectory = $@"{Path.GetDirectoryName(typeof(MatrixMultiplyShould).Assembly.Location)}\Tensors\LinearAlgebra\Examples\";
    }

    private static Array MatrixMatrixTest<TNumber>(TNumber[,] left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftTensor = Tensor.FromArray<TNumber>(left).ToMatrix();
        var rightTensor = Tensor.FromArray<TNumber>(right).ToMatrix();
        leftTensor.To(device);
        rightTensor.To(device);

        var result = leftTensor.MatrixMultiply(rightTensor);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    private static void MatrixMatrixTest<TNumber>(Matrix<TNumber> left, Matrix<TNumber> right, Matrix<TNumber> expectedResult, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        left.To(device);
        right.To(device);

        var result = left.MatrixMultiply(right);

        result.To<CpuComputeDevice>();

        result.Should().HaveEquivalentElements(expectedResult.ToArray());
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
    public void ReturnExpectedResult_GivenExample1(IDevice device)
    {
        var tensors = Tensor.LoadSafeTensors<long>($"{_safetensorsLoadDirectory}matmul_1.safetensors");
        var left = tensors["left"].ToMatrix();
        var right = tensors["right"].ToMatrix();
        var expected = tensors["result"].ToMatrix();

        MatrixMatrixTest(left, right, expected, device);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenExample2(IDevice device)
    {
        var tensors = Tensor.LoadSafeTensors<long>($"{_safetensorsLoadDirectory}matmul_2.safetensors");
        var left = tensors["left"].ToMatrix();
        var right = tensors["right"].ToMatrix();
        var expected = tensors["result"].ToMatrix();

        MatrixMatrixTest(left, right, expected, device);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenExample3(IDevice device)
    {
        var tensors = Tensor.LoadSafeTensors<long>($"{_safetensorsLoadDirectory}matmul_3.safetensors");
        var left = tensors["left"].ToMatrix();
        var right = tensors["right"].ToMatrix();
        var expected = tensors["result"].ToMatrix();

        MatrixMatrixTest(left, right, expected, device);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenExample4(IDevice device)
    {
        var tensors = Tensor.LoadSafeTensors<long>($"{_safetensorsLoadDirectory}matmul_4.safetensors");
        var left = tensors["left"].ToMatrix();
        var right = tensors["right"].ToMatrix();
        var expected = tensors["result"].ToMatrix();

        MatrixMatrixTest(left, right, expected, device);
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenExample5(IDevice device)
    {
        var tensors = Tensor.LoadSafeTensors<long>($"{_safetensorsLoadDirectory}matmul_5.safetensors");
        var left = tensors["left"].ToMatrix();
        var right = tensors["right"].ToMatrix();
        var expected = tensors["result"].ToMatrix();

        MatrixMatrixTest(left, right, expected, device);
    }
}