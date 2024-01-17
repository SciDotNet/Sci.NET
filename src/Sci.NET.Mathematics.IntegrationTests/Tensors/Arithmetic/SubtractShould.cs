// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Arithmetic;

public class SubtractShould : IntegrationTestBase, IArithmeticTests
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarsAndScalar(IDevice device)
    {
        ScalarScalarTest<BFloat16>(2, 1, device).Should().Be(1);
        ScalarScalarTest<float>(2, 1, device).Should().Be(1);
        ScalarScalarTest<double>(2, 1, device).Should().Be(1);
        ScalarScalarTest<byte>(2, 1, device).Should().Be(1);
        ScalarScalarTest<sbyte>(2, 1, device).Should().Be(1);
        ScalarScalarTest<ushort>(2, 1, device).Should().Be(1);
        ScalarScalarTest<short>(2, 1, device).Should().Be(1);
        ScalarScalarTest<uint>(2, 1, device).Should().Be(1);
        ScalarScalarTest<int>(2, 1, device).Should().Be(1);
        ScalarScalarTest<long>(2, 1, device).Should().Be(1);
        ScalarScalarTest<ulong>(2, 1, device).Should().Be(1);
    }

    private static TNumber ScalarScalarTest<TNumber>(TNumber left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = new Scalar<TNumber>(left);
        var rightScalar = new Scalar<TNumber>(right);
        leftScalar.To(device);
        rightScalar.To(device);

        var result = leftScalar.Subtract(rightScalar);

        result.To<CpuComputeDevice>();
        return result.Value;
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarAndVector(IDevice device)
    {
        ScalarVectorTest<BFloat16>(5, new BFloat16[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new BFloat16[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest<float>(5, new float[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new float[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest<double>(5, new double[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new double[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest<byte>(5, new byte[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new byte[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest<sbyte>(5, new sbyte[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new sbyte[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest<ushort>(5, new ushort[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new ushort[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest<short>(5, new short[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new short[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest<uint>(5, new uint[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new uint[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest<int>(5, new int[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new int[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest<ulong>(5, new ulong[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new ulong[] { 4, 3, 2, 1, 0 });
        ScalarVectorTest<long>(5, new long[] { 1, 2, 3, 4, 5 }, device).Should().BeEquivalentTo(new long[] { 4, 3, 2, 1, 0 });
    }

    private static Array ScalarVectorTest<TNumber>(TNumber left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = new Scalar<TNumber>(left);
        var rightVector = Tensor.FromArray<TNumber>(right).ToVector();
        leftScalar.To(device);
        rightVector.To(device);

        var result = leftScalar.Subtract(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarAndMatrix(IDevice device)
    {
        ScalarMatrixTest<BFloat16>(5, new BFloat16[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new BFloat16[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest<float>(5, new float[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new float[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest<double>(5, new double[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new double[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest<byte>(5, new byte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new byte[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest<sbyte>(5, new sbyte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new sbyte[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest<ushort>(5, new ushort[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ushort[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest<short>(5, new short[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new short[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest<uint>(5, new uint[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new uint[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest<int>(5, new int[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new int[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest<ulong>(5, new ulong[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ulong[,] { { 4, 3 }, { 2, 1 } });
        ScalarMatrixTest<long>(5, new long[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new long[,] { { 4, 3 }, { 2, 1 } });
    }

    private static Array ScalarMatrixTest<TNumber>(TNumber left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = new Scalar<TNumber>(left);
        var rightMatrix = Tensor.FromArray<TNumber>(right).ToMatrix();
        leftScalar.To(device);
        rightMatrix.To(device);

        var result = leftScalar.Subtract(rightMatrix);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarTensor(IDevice device)
    {
        ScalarTensorTest<BFloat16>(5, new BFloat16[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new BFloat16[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest<float>(5, new float[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new float[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest<double>(5, new double[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new double[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest<byte>(5, new byte[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new byte[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest<sbyte>(5, new sbyte[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new sbyte[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest<ushort>(5, new ushort[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new ushort[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest<short>(5, new short[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new short[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest<uint>(5, new uint[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new uint[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest<int>(5, new int[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new int[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest<ulong>(5, new ulong[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new ulong[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
        ScalarTensorTest<long>(5, new long[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new long[,,,] { { { { 4, 3 }, { 2, 1 } }, { { 4, 3 }, { 2, 1 } } } });
    }

    private static Array ScalarTensorTest<TNumber>(TNumber left, TNumber[,,,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = new Scalar<TNumber>(left);
        var rightTensor = Tensor.FromArray<TNumber>(right).ToTensor();
        leftScalar.To(device);
        rightTensor.To(device);

        var result = leftScalar.Subtract(rightTensor);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndScalar(IDevice device)
    {
        VectorScalarTest<BFloat16>(new BFloat16[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new BFloat16[] { 5, 4, 3, 2, 1 });
        VectorScalarTest<float>(new float[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new float[] { 5, 4, 3, 2, 1 });
        VectorScalarTest<double>(new double[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new double[] { 5, 4, 3, 2, 1 });
        VectorScalarTest<byte>(new byte[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new byte[] { 5, 4, 3, 2, 1 });
        VectorScalarTest<sbyte>(new sbyte[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new sbyte[] { 5, 4, 3, 2, 1 });
        VectorScalarTest<ushort>(new ushort[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new ushort[] { 5, 4, 3, 2, 1 });
        VectorScalarTest<short>(new short[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new short[] { 5, 4, 3, 2, 1 });
        VectorScalarTest<uint>(new uint[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new uint[] { 5, 4, 3, 2, 1 });
        VectorScalarTest<int>(new int[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new int[] { 5, 4, 3, 2, 1 });
        VectorScalarTest<ulong>(new ulong[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new ulong[] { 5, 4, 3, 2, 1 });
        VectorScalarTest<long>(new long[] { 7, 6, 5, 4, 3 }, 2, device).Should().BeEquivalentTo(new long[] { 5, 4, 3, 2, 1 });
    }

    private static Array VectorScalarTest<TNumber>(TNumber[] left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftVector = Tensor.FromArray<TNumber>(left).ToVector();
        var rightScalar = new Scalar<TNumber>(right);
        leftVector.To(device);
        rightScalar.To(device);

        var result = leftVector.Subtract(rightScalar);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndVector(IDevice device)
    {
        VectorVectorTest<BFloat16>(new BFloat16[] { 7, 6, 5, 4, 3 }, new BFloat16[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new BFloat16[] { 5, 4, 3, 2, 1 });
        VectorVectorTest<float>(new float[] { 7, 6, 5, 4, 3 }, new float[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new float[] { 5, 4, 3, 2, 1 });
        VectorVectorTest<double>(new double[] { 7, 6, 5, 4, 3 }, new double[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new double[] { 5, 4, 3, 2, 1 });
        VectorVectorTest<byte>(new byte[] { 7, 6, 5, 4, 3 }, new byte[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new byte[] { 5, 4, 3, 2, 1 });
        VectorVectorTest<sbyte>(new sbyte[] { 7, 6, 5, 4, 3 }, new sbyte[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new sbyte[] { 5, 4, 3, 2, 1 });
        VectorVectorTest<ushort>(new ushort[] { 7, 6, 5, 4, 3 }, new ushort[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new ushort[] { 5, 4, 3, 2, 1 });
        VectorVectorTest<short>(new short[] { 7, 6, 5, 4, 3 }, new short[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new short[] { 5, 4, 3, 2, 1 });
        VectorVectorTest<uint>(new uint[] { 7, 6, 5, 4, 3 }, new uint[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new uint[] { 5, 4, 3, 2, 1 });
        VectorVectorTest<int>(new int[] { 7, 6, 5, 4, 3 }, new int[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new int[] { 5, 4, 3, 2, 1 });
        VectorVectorTest<ulong>(new ulong[] { 7, 6, 5, 4, 3 }, new ulong[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new ulong[] { 5, 4, 3, 2, 1 });
        VectorVectorTest<long>(new long[] { 7, 6, 5, 4, 3 }, new long[] { 2, 2, 2, 2, 2 }, device).Should().BeEquivalentTo(new long[] { 5, 4, 3, 2, 1 });
    }

    private static Array VectorVectorTest<TNumber>(TNumber[] left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftVector = Tensor.FromArray<TNumber>(left).ToVector();
        var rightVector = Tensor.FromArray<TNumber>(right).ToVector();
        leftVector.To(device);
        rightVector.To(device);

        var result = leftVector.Subtract(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndMatrix(IDevice device)
    {
        VectorMatrixTest<BFloat16>(new BFloat16[] { 7, 6 }, new BFloat16[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new BFloat16[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest<float>(new float[] { 7, 6 }, new float[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new float[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest<double>(new double[] { 7, 6 }, new double[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new double[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest<byte>(new byte[] { 7, 6 }, new byte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new byte[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest<sbyte>(new sbyte[] { 7, 6 }, new sbyte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new sbyte[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest<ushort>(new ushort[] { 7, 6 }, new ushort[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ushort[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest<short>(new short[] { 7, 6 }, new short[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new short[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest<uint>(new uint[] { 7, 6 }, new uint[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new uint[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest<int>(new int[] { 7, 6 }, new int[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new int[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest<ulong>(new ulong[] { 7, 6 }, new ulong[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ulong[,] { { 6, 4 }, { 4, 2 } });
        VectorMatrixTest<long>(new long[] { 7, 6 }, new long[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new long[,] { { 6, 4 }, { 4, 2 } });
    }

    private static Array VectorMatrixTest<TNumber>(TNumber[] left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftVector = Tensor.FromArray<TNumber>(left).ToVector();
        var rightMatrix = Tensor.FromArray<TNumber>(right).ToMatrix();
        leftVector.To(device);
        rightMatrix.To(device);

        var result = leftVector.Subtract(rightMatrix);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndTensor(IDevice device)
    {
        VectorTensorTest<BFloat16>(new BFloat16[] { 7, 6 }, new BFloat16[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new BFloat16[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest<float>(new float[] { 7, 6 }, new float[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new float[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest<double>(new double[] { 7, 6 }, new double[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new double[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest<byte>(new byte[] { 7, 6 }, new byte[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new byte[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest<sbyte>(new sbyte[] { 7, 6 }, new sbyte[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new sbyte[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest<ushort>(new ushort[] { 7, 6 }, new ushort[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new ushort[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest<short>(new short[] { 7, 6 }, new short[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new short[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest<uint>(new uint[] { 7, 6 }, new uint[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new uint[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest<int>(new int[] { 7, 6 }, new int[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new int[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest<ulong>(new ulong[] { 7, 6 }, new ulong[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new ulong[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
        VectorTensorTest<long>(new long[] { 7, 6 }, new long[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, device).Should().BeEquivalentTo(new long[,,,] { { { { 6, 4 }, { 4, 2 } }, { { 6, 4 }, { 4, 2 } } } });
    }

    private static Array VectorTensorTest<TNumber>(TNumber[] left, TNumber[,,,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftVector = Tensor.FromArray<TNumber>(left).ToVector();
        var rightTensor = Tensor.FromArray<TNumber>(right).ToTensor();
        leftVector.To(device);
        rightTensor.To(device);

        var result = leftVector.Subtract(rightTensor);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndScalar(IDevice device)
    {
        MatrixScalarTest<BFloat16>(new BFloat16[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new BFloat16[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest<float>(new float[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new float[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest<double>(new double[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new double[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest<byte>(new byte[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new byte[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest<sbyte>(new sbyte[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new sbyte[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest<ushort>(new ushort[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new ushort[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest<short>(new short[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new short[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest<uint>(new uint[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new uint[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest<int>(new int[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new int[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest<ulong>(new ulong[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new ulong[,] { { 1, 3 }, { 5, 7 } });
        MatrixScalarTest<long>(new long[,] { { 2, 4 }, { 6, 8 } }, 1, device).Should().BeEquivalentTo(new long[,] { { 1, 3 }, { 5, 7 } });
    }

    private static Array MatrixScalarTest<TNumber>(TNumber[,] left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMatrix = Tensor.FromArray<TNumber>(left).ToMatrix();
        var rightScalar = new Scalar<TNumber>(right);
        leftMatrix.To(device);
        rightScalar.To(device);

        var result = leftMatrix.Subtract(rightScalar);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndVector(IDevice device)
    {
        MatrixVectorTest<BFloat16>(new BFloat16[,] { { 2, 4 }, { 6, 8 } }, new BFloat16[] { 1, 2 }, device).Should().BeEquivalentTo(new BFloat16[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest<float>(new float[,] { { 2, 4 }, { 6, 8 } }, new float[] { 1, 2 }, device).Should().BeEquivalentTo(new float[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest<double>(new double[,] { { 2, 4 }, { 6, 8 } }, new double[] { 1, 2 }, device).Should().BeEquivalentTo(new double[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest<byte>(new byte[,] { { 2, 4 }, { 6, 8 } }, new byte[] { 1, 2 }, device).Should().BeEquivalentTo(new byte[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest<sbyte>(new sbyte[,] { { 2, 4 }, { 6, 8 } }, new sbyte[] { 1, 2 }, device).Should().BeEquivalentTo(new sbyte[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest<ushort>(new ushort[,] { { 2, 4 }, { 6, 8 } }, new ushort[] { 1, 2 }, device).Should().BeEquivalentTo(new ushort[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest<short>(new short[,] { { 2, 4 }, { 6, 8 } }, new short[] { 1, 2 }, device).Should().BeEquivalentTo(new short[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest<uint>(new uint[,] { { 2, 4 }, { 6, 8 } }, new uint[] { 1, 2 }, device).Should().BeEquivalentTo(new uint[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest<int>(new int[,] { { 2, 4 }, { 6, 8 } }, new int[] { 1, 2 }, device).Should().BeEquivalentTo(new int[,] { { 1, 2 }, { 5, 6 } });
        MatrixVectorTest<ulong>(new ulong[,] { { 2, 4 }, { 6, 8 } }, new ulong[] { 1, 2 }, device).Should().BeEquivalentTo(new ulong[,] { { 1, 2 }, { 5, 6 } });
    }

    private static Array MatrixVectorTest<TNumber>(TNumber[,] left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMatrix = Tensor.FromArray<TNumber>(left).ToMatrix();
        var rightVector = Tensor.FromArray<TNumber>(right).ToVector();
        leftMatrix.To(device);
        rightVector.To(device);

        var result = leftMatrix.Subtract(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndMatrix(IDevice device)
    {
        MatrixMatrixTest<BFloat16>(new BFloat16[,] { { 2, 4 }, { 6, 8 } }, new BFloat16[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new BFloat16[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest<float>(new float[,] { { 2, 4 }, { 6, 8 } }, new float[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new float[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest<double>(new double[,] { { 2, 4 }, { 6, 8 } }, new double[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new double[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest<byte>(new byte[,] { { 2, 4 }, { 6, 8 } }, new byte[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new byte[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest<sbyte>(new sbyte[,] { { 2, 4 }, { 6, 8 } }, new sbyte[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new sbyte[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest<ushort>(new ushort[,] { { 2, 4 }, { 6, 8 } }, new ushort[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new ushort[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest<short>(new short[,] { { 2, 4 }, { 6, 8 } }, new short[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new short[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest<uint>(new uint[,] { { 2, 4 }, { 6, 8 } }, new uint[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new uint[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest<int>(new int[,] { { 2, 4 }, { 6, 8 } }, new int[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new int[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest<ulong>(new ulong[,] { { 2, 4 }, { 6, 8 } }, new ulong[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new ulong[,] { { 2, 3 }, { 6, 7 } });
        MatrixMatrixTest<long>(new long[,] { { 2, 4 }, { 6, 8 } }, new long[,] { { 0, 1 }, { 0, 1 } }, device).Should().BeEquivalentTo(new long[,] { { 2, 3 }, { 6, 7 } });
    }

    private static Array MatrixMatrixTest<TNumber>(TNumber[,] left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMatrix = Tensor.FromArray<TNumber>(left).ToMatrix();
        var rightMatrix = Tensor.FromArray<TNumber>(right).ToMatrix();
        leftMatrix.To(device);
        rightMatrix.To(device);

        var result = leftMatrix.Subtract(rightMatrix);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndTensor(IDevice device)
    {
        MatrixTensorTest<BFloat16>(new BFloat16[,] { { 2, 4 }, { 6, 8 } }, new BFloat16[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new BFloat16[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest<float>(new float[,] { { 2, 4 }, { 6, 8 } }, new float[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new float[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest<double>(new double[,] { { 2, 4 }, { 6, 8 } }, new double[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new double[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest<byte>(new byte[,] { { 2, 4 }, { 6, 8 } }, new byte[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new byte[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest<sbyte>(new sbyte[,] { { 2, 4 }, { 6, 8 } }, new sbyte[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new sbyte[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest<ushort>(new ushort[,] { { 2, 4 }, { 6, 8 } }, new ushort[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new ushort[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest<short>(new short[,] { { 2, 4 }, { 6, 8 } }, new short[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new short[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest<uint>(new uint[,] { { 2, 4 }, { 6, 8 } }, new uint[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new uint[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest<int>(new int[,] { { 2, 4 }, { 6, 8 } }, new int[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new int[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest<ulong>(new ulong[,] { { 2, 4 }, { 6, 8 } }, new ulong[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new ulong[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
        MatrixTensorTest<long>(new long[,] { { 2, 4 }, { 6, 8 } }, new long[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new long[,,,] { { { { 2, 3 }, { 4, 5 } }, { { 2, 3 }, { 4, 5 } } } });
    }

    private static Array MatrixTensorTest<TNumber>(TNumber[,] left, TNumber[,,,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftMatrix = Tensor.FromArray<TNumber>(left).ToMatrix();
        var rightTensor = Tensor.FromArray<TNumber>(right).ToTensor();
        leftMatrix.To(device);
        rightTensor.To(device);

        var result = leftMatrix.Subtract(rightTensor);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorAndScalar(IDevice device)
    {
        TensorScalarTest<BFloat16>(new BFloat16[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new BFloat16[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest<float>(new float[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new float[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest<double>(new double[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new double[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest<byte>(new byte[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new byte[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest<sbyte>(new sbyte[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new sbyte[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest<ushort>(new ushort[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new ushort[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest<short>(new short[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new short[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest<uint>(new uint[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new uint[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest<int>(new int[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new int[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest<ulong>(new ulong[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new ulong[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
        TensorScalarTest<long>(new long[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 1, 2 }, { 3, 4 } } } }, 1, device).Should().BeEquivalentTo(new long[,,,] { { { { 0, 1 }, { 2, 3 } }, { { 0, 1 }, { 2, 3 } } } });
    }

    private static Array TensorScalarTest<TNumber>(TNumber[,,,] left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftTensor = Tensor.FromArray<TNumber>(left).ToTensor();
        var rightScalar = new Scalar<TNumber>(right);
        leftTensor.To(device);
        rightScalar.To(device);

        var result = leftTensor.Subtract(rightScalar);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorAndVector(IDevice device)
    {
        TensorVectorTest<BFloat16>(new BFloat16[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new BFloat16[] { 0, 1 }, device).Should().BeEquivalentTo(new BFloat16[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest<float>(new float[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new float[] { 0, 1 }, device).Should().BeEquivalentTo(new float[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest<double>(new double[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new double[] { 0, 1 }, device).Should().BeEquivalentTo(new double[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest<byte>(new byte[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new byte[] { 0, 1 }, device).Should().BeEquivalentTo(new byte[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest<sbyte>(new sbyte[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new sbyte[] { 0, 1 }, device).Should().BeEquivalentTo(new sbyte[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest<ushort>(new ushort[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new ushort[] { 0, 1 }, device).Should().BeEquivalentTo(new ushort[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest<short>(new short[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new short[] { 0, 1 }, device).Should().BeEquivalentTo(new short[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest<uint>(new uint[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new uint[] { 0, 1 }, device).Should().BeEquivalentTo(new uint[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest<int>(new int[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new int[] { 0, 1 }, device).Should().BeEquivalentTo(new int[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest<ulong>(new ulong[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new ulong[] { 0, 1 }, device).Should().BeEquivalentTo(new ulong[,,,] { { { { 1, 1 }, { 3, 3 } } } });
        TensorVectorTest<long>(new long[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new long[] { 0, 1 }, device).Should().BeEquivalentTo(new long[,,,] { { { { 1, 1 }, { 3, 3 } } } });
    }

    private static Array TensorVectorTest<TNumber>(TNumber[,,,] left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftTensor = Tensor.FromArray<TNumber>(left).ToTensor();
        var rightVector = Tensor.FromArray<TNumber>(right).ToVector();
        leftTensor.To(device);
        rightVector.To(device);

        var result = leftTensor.Subtract(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorAndMatrix(IDevice device)
    {
        TensorMatrixTest<BFloat16>(new BFloat16[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new BFloat16[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new BFloat16[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest<float>(new float[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new float[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new float[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest<double>(new double[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new double[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new double[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest<byte>(new byte[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new byte[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new byte[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest<sbyte>(new sbyte[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new sbyte[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new sbyte[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest<ushort>(new ushort[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new ushort[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new ushort[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest<short>(new short[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new short[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new short[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest<uint>(new uint[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new uint[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new uint[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest<int>(new int[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new int[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new int[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest<ulong>(new ulong[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new ulong[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new ulong[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorMatrixTest<long>(new long[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new long[,] { { 0, 1 }, { 2, 3 } }, device).Should().BeEquivalentTo(new long[,,,] { { { { 1, 1 }, { 1, 1 } } } });
    }

    private static Array TensorMatrixTest<TNumber>(TNumber[,,,] left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftTensor = Tensor.FromArray<TNumber>(left).ToTensor();
        var rightMatrix = Tensor.FromArray<TNumber>(right).ToMatrix();
        leftTensor.To(device);
        rightMatrix.To(device);

        var result = leftTensor.Subtract(rightMatrix);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorAndTensor(IDevice device)
    {
        TensorTensorTest<BFloat16>(new BFloat16[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new BFloat16[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new BFloat16[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest<float>(new float[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new float[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new float[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest<double>(new double[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new double[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new double[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest<byte>(new byte[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new byte[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new byte[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest<sbyte>(new sbyte[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new sbyte[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new sbyte[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest<ushort>(new ushort[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new ushort[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new ushort[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest<short>(new short[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new short[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new short[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest<uint>(new uint[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new uint[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new uint[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest<int>(new int[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new int[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new int[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest<ulong>(new ulong[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new ulong[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new ulong[,,,] { { { { 1, 1 }, { 1, 1 } } } });
        TensorTensorTest<long>(new long[,,,] { { { { 1, 2 }, { 3, 4 } } } }, new long[,,,] { { { { 0, 1 }, { 2, 3 } } } }, device).Should().BeEquivalentTo(new long[,,,] { { { { 1, 1 }, { 1, 1 } } } });
    }

    private static Array TensorTensorTest<TNumber>(TNumber[,,,] left, TNumber[,,,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftTensor = Tensor.FromArray<TNumber>(left).ToTensor();
        var rightTensor = Tensor.FromArray<TNumber>(right).ToTensor();
        leftTensor.To(device);
        rightTensor.To(device);

        var result = leftTensor.Subtract(rightTensor);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }
}