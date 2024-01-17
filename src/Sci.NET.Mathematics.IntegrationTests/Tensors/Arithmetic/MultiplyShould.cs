// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Arithmetic;

public class MultiplyShould : IntegrationTestBase, IArithmeticTests
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarsAndScalar(IDevice device)
    {
        ScalarScalarTest<BFloat16>(2, 2, device).Should().Be(4);
        ScalarScalarTest<float>(2, 2, device).Should().Be(4);
        ScalarScalarTest<double>(2, 2, device).Should().Be(4);
        ScalarScalarTest<byte>(2, 2, device).Should().Be(4);
        ScalarScalarTest<sbyte>(2, 2, device).Should().Be(4);
        ScalarScalarTest<ushort>(2, 2, device).Should().Be(4);
        ScalarScalarTest<short>(2, 2, device).Should().Be(4);
        ScalarScalarTest<uint>(2, 2, device).Should().Be(4);
        ScalarScalarTest<int>(2, 2, device).Should().Be(4);
        ScalarScalarTest<ulong>(2, 2, device).Should().Be(4);
        ScalarScalarTest<long>(2, 2, device).Should().Be(4);
    }

    private static TNumber ScalarScalarTest<TNumber>(TNumber left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftScalar = new Scalar<TNumber>(left);
        using var rightScalar = new Scalar<TNumber>(right);

        leftScalar.To(device);
        rightScalar.To(device);

        using var result = leftScalar.Multiply(rightScalar);

        return result.Value;
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarAndVector(IDevice device)
    {
        ScalarVectorTest<BFloat16>(2, new BFloat16[] { 1, 2 }, device).Should().BeEquivalentTo(new BFloat16[] { 2, 4 });
        ScalarVectorTest<float>(2, new float[] { 1, 2 }, device).Should().BeEquivalentTo(new float[] { 2, 4 });
        ScalarVectorTest<double>(2, new double[] { 1, 2 }, device).Should().BeEquivalentTo(new double[] { 2, 4 });
        ScalarVectorTest<byte>(2, new byte[] { 1, 2 }, device).Should().BeEquivalentTo(new byte[] { 2, 4 });
        ScalarVectorTest<sbyte>(2, new sbyte[] { 1, 2 }, device).Should().BeEquivalentTo(new sbyte[] { 2, 4 });
        ScalarVectorTest<ushort>(2, new ushort[] { 1, 2 }, device).Should().BeEquivalentTo(new ushort[] { 2, 4 });
        ScalarVectorTest<short>(2, new short[] { 1, 2 }, device).Should().BeEquivalentTo(new short[] { 2, 4 });
        ScalarVectorTest<uint>(2, new uint[] { 1, 2 }, device).Should().BeEquivalentTo(new uint[] { 2, 4 });
        ScalarVectorTest<int>(2, new int[] { 1, 2 }, device).Should().BeEquivalentTo(new int[] { 2, 4 });
        ScalarVectorTest<ulong>(2, new ulong[] { 1, 2 }, device).Should().BeEquivalentTo(new ulong[] { 2, 4 });
        ScalarVectorTest<long>(2, new long[] { 1, 2 }, device).Should().BeEquivalentTo(new long[] { 2, 4 });
    }

    private static Array ScalarVectorTest<TNumber>(TNumber left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftScalar = new Scalar<TNumber>(left);
        using var rightVector = Tensor.FromArray<TNumber>(right).ToVector();

        leftScalar.To(device);
        rightVector.To(device);

        using var result = leftScalar.Multiply(rightVector);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarAndMatrix(IDevice device)
    {
        ScalarMatrixTest<BFloat16>(2, new BFloat16[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new BFloat16[,] { { 2, 4 }, { 6, 8 } });
        ScalarMatrixTest<float>(2, new float[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new float[,] { { 2, 4 }, { 6, 8 } });
        ScalarMatrixTest<double>(2, new double[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new double[,] { { 2, 4 }, { 6, 8 } });
        ScalarMatrixTest<byte>(2, new byte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new byte[,] { { 2, 4 }, { 6, 8 } });
        ScalarMatrixTest<sbyte>(2, new sbyte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new sbyte[,] { { 2, 4 }, { 6, 8 } });
        ScalarMatrixTest<ushort>(2, new ushort[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ushort[,] { { 2, 4 }, { 6, 8 } });
        ScalarMatrixTest<short>(2, new short[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new short[,] { { 2, 4 }, { 6, 8 } });
        ScalarMatrixTest<uint>(2, new uint[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new uint[,] { { 2, 4 }, { 6, 8 } });
        ScalarMatrixTest<int>(2, new int[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new int[,] { { 2, 4 }, { 6, 8 } });
        ScalarMatrixTest<ulong>(2, new ulong[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ulong[,] { { 2, 4 }, { 6, 8 } });
        ScalarMatrixTest<long>(2, new long[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new long[,] { { 2, 4 }, { 6, 8 } });
    }

    private static Array ScalarMatrixTest<TNumber>(TNumber left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftScalar = new Scalar<TNumber>(left);
        using var rightMatrix = Tensor.FromArray<TNumber>(right).ToMatrix();

        leftScalar.To(device);
        rightMatrix.To(device);

        using var result = leftScalar.Multiply(rightMatrix);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarTensor(IDevice device)
    {
        ScalarTensorTest<BFloat16>(2, new BFloat16[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new BFloat16[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        ScalarTensorTest<float>(2, new float[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new float[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        ScalarTensorTest<double>(2, new double[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new double[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        ScalarTensorTest<byte>(2, new byte[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new byte[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        ScalarTensorTest<sbyte>(2, new sbyte[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new sbyte[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        ScalarTensorTest<ushort>(2, new ushort[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new ushort[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        ScalarTensorTest<short>(2, new short[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new short[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        ScalarTensorTest<uint>(2, new uint[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new uint[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        ScalarTensorTest<int>(2, new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new int[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        ScalarTensorTest<ulong>(2, new ulong[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new ulong[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        ScalarTensorTest<long>(2, new long[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new long[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
    }

    private static Array ScalarTensorTest<TNumber>(TNumber left, TNumber[,,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftScalar = new Scalar<TNumber>(left);
        using var rightTensor = Tensor.FromArray<TNumber>(right).ToTensor();

        leftScalar.To(device);
        rightTensor.To(device);

        using var result = leftScalar.Multiply(rightTensor);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndScalar(IDevice device)
    {
        VectorScalarTest<BFloat16>(new BFloat16[] { 1, 2 }, 2, device).Should().BeEquivalentTo(new BFloat16[] { 2, 4 });
        VectorScalarTest<float>(new float[] { 1, 2 }, 2, device).Should().BeEquivalentTo(new float[] { 2, 4 });
        VectorScalarTest<double>(new double[] { 1, 2 }, 2, device).Should().BeEquivalentTo(new double[] { 2, 4 });
        VectorScalarTest<byte>(new byte[] { 1, 2 }, 2, device).Should().BeEquivalentTo(new byte[] { 2, 4 });
        VectorScalarTest<sbyte>(new sbyte[] { 1, 2 }, 2, device).Should().BeEquivalentTo(new sbyte[] { 2, 4 });
        VectorScalarTest<ushort>(new ushort[] { 1, 2 }, 2, device).Should().BeEquivalentTo(new ushort[] { 2, 4 });
        VectorScalarTest<short>(new short[] { 1, 2 }, 2, device).Should().BeEquivalentTo(new short[] { 2, 4 });
        VectorScalarTest<uint>(new uint[] { 1, 2 }, 2, device).Should().BeEquivalentTo(new uint[] { 2, 4 });
        VectorScalarTest<int>(new int[] { 1, 2 }, 2, device).Should().BeEquivalentTo(new int[] { 2, 4 });
        VectorScalarTest<ulong>(new ulong[] { 1, 2 }, 2, device).Should().BeEquivalentTo(new ulong[] { 2, 4 });
        VectorScalarTest<long>(new long[] { 1, 2 }, 2, device).Should().BeEquivalentTo(new long[] { 2, 4 });
    }

    private static Array VectorScalarTest<TNumber>(TNumber[] left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftVector = Tensor.FromArray<TNumber>(left).ToVector();
        using var rightScalar = new Scalar<TNumber>(right);

        leftVector.To(device);
        rightScalar.To(device);

        using var result = leftVector.Multiply(rightScalar);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndVector(IDevice device)
    {
        VectorVectorTest<BFloat16>(new BFloat16[] { 1, 2 }, new BFloat16[] { 1, 2 }, device).Should().BeEquivalentTo(new BFloat16[] { 1, 4 });
        VectorVectorTest<float>(new float[] { 1, 2 }, new float[] { 1, 2 }, device).Should().BeEquivalentTo(new float[] { 1, 4 });
        VectorVectorTest<double>(new double[] { 1, 2 }, new double[] { 1, 2 }, device).Should().BeEquivalentTo(new double[] { 1, 4 });
        VectorVectorTest<byte>(new byte[] { 1, 2 }, new byte[] { 1, 2 }, device).Should().BeEquivalentTo(new byte[] { 1, 4 });
        VectorVectorTest<sbyte>(new sbyte[] { 1, 2 }, new sbyte[] { 1, 2 }, device).Should().BeEquivalentTo(new sbyte[] { 1, 4 });
        VectorVectorTest<ushort>(new ushort[] { 1, 2 }, new ushort[] { 1, 2 }, device).Should().BeEquivalentTo(new ushort[] { 1, 4 });
        VectorVectorTest<short>(new short[] { 1, 2 }, new short[] { 1, 2 }, device).Should().BeEquivalentTo(new short[] { 1, 4 });
        VectorVectorTest<uint>(new uint[] { 1, 2 }, new uint[] { 1, 2 }, device).Should().BeEquivalentTo(new uint[] { 1, 4 });
        VectorVectorTest<int>(new int[] { 1, 2 }, new int[] { 1, 2 }, device).Should().BeEquivalentTo(new int[] { 1, 4 });
        VectorVectorTest<ulong>(new ulong[] { 1, 2 }, new ulong[] { 1, 2 }, device).Should().BeEquivalentTo(new ulong[] { 1, 4 });
        VectorVectorTest<long>(new long[] { 1, 2 }, new long[] { 1, 2 }, device).Should().BeEquivalentTo(new long[] { 1, 4 });
    }

    private static Array VectorVectorTest<TNumber>(TNumber[] left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftVector = Tensor.FromArray<TNumber>(left).ToVector();
        using var rightVector = Tensor.FromArray<TNumber>(right).ToVector();

        leftVector.To(device);
        rightVector.To(device);

        using var result = leftVector.Multiply(rightVector);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndMatrix(IDevice device)
    {
        VectorMatrixTest<BFloat16>(new BFloat16[] { 1, 2 }, new BFloat16[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new BFloat16[,] { { 1, 4 }, { 3, 8 } });
        VectorMatrixTest<float>(new float[] { 1, 2 }, new float[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new float[,] { { 1, 4 }, { 3, 8 } });
        VectorMatrixTest<double>(new double[] { 1, 2 }, new double[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new double[,] { { 1, 4 }, { 3, 8 } });
        VectorMatrixTest<byte>(new byte[] { 1, 2 }, new byte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new byte[,] { { 1, 4 }, { 3, 8 } });
        VectorMatrixTest<sbyte>(new sbyte[] { 1, 2 }, new sbyte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new sbyte[,] { { 1, 4 }, { 3, 8 } });
        VectorMatrixTest<ushort>(new ushort[] { 1, 2 }, new ushort[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ushort[,] { { 1, 4 }, { 3, 8 } });
        VectorMatrixTest<short>(new short[] { 1, 2 }, new short[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new short[,] { { 1, 4 }, { 3, 8 } });
        VectorMatrixTest<uint>(new uint[] { 1, 2 }, new uint[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new uint[,] { { 1, 4 }, { 3, 8 } });
        VectorMatrixTest<int>(new int[] { 1, 2 }, new int[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new int[,] { { 1, 4 }, { 3, 8 } });
        VectorMatrixTest<ulong>(new ulong[] { 1, 2 }, new ulong[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ulong[,] { { 1, 4 }, { 3, 8 } });
        VectorMatrixTest<long>(new long[] { 1, 2 }, new long[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new long[,] { { 1, 4 }, { 3, 8 } });
    }

    private static Array VectorMatrixTest<TNumber>(TNumber[] left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftVector = Tensor.FromArray<TNumber>(left).ToVector();
        using var rightMatrix = Tensor.FromArray<TNumber>(right).ToMatrix();

        leftVector.To(device);
        rightMatrix.To(device);

        using var result = leftVector.Multiply(rightMatrix);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndTensor(IDevice device)
    {
        VectorTensorTest<BFloat16>(new BFloat16[] { 1, 2 }, new BFloat16[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new BFloat16[,,] { { { 1, 4 }, { 3, 8 } }, { { 5, 12 }, { 7, 16 } } });
        VectorTensorTest<float>(new float[] { 1, 2 }, new float[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new float[,,] { { { 1, 4 }, { 3, 8 } }, { { 5, 12 }, { 7, 16 } } });
        VectorTensorTest<double>(new double[] { 1, 2 }, new double[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new double[,,] { { { 1, 4 }, { 3, 8 } }, { { 5, 12 }, { 7, 16 } } });
        VectorTensorTest<byte>(new byte[] { 1, 2 }, new byte[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new byte[,,] { { { 1, 4 }, { 3, 8 } }, { { 5, 12 }, { 7, 16 } } });
        VectorTensorTest<sbyte>(new sbyte[] { 1, 2 }, new sbyte[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new sbyte[,,] { { { 1, 4 }, { 3, 8 } }, { { 5, 12 }, { 7, 16 } } });
        VectorTensorTest<ushort>(new ushort[] { 1, 2 }, new ushort[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new ushort[,,] { { { 1, 4 }, { 3, 8 } }, { { 5, 12 }, { 7, 16 } } });
        VectorTensorTest<short>(new short[] { 1, 2 }, new short[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new short[,,] { { { 1, 4 }, { 3, 8 } }, { { 5, 12 }, { 7, 16 } } });
        VectorTensorTest<uint>(new uint[] { 1, 2 }, new uint[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new uint[,,] { { { 1, 4 }, { 3, 8 } }, { { 5, 12 }, { 7, 16 } } });
        VectorTensorTest<int>(new int[] { 1, 2 }, new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new int[,,] { { { 1, 4 }, { 3, 8 } }, { { 5, 12 }, { 7, 16 } } });
        VectorTensorTest<ulong>(new ulong[] { 1, 2 }, new ulong[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new ulong[,,] { { { 1, 4 }, { 3, 8 } }, { { 5, 12 }, { 7, 16 } } });
        VectorTensorTest<long>(new long[] { 1, 2 }, new long[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new long[,,] { { { 1, 4 }, { 3, 8 } }, { { 5, 12 }, { 7, 16 } } });
    }

    private static Array VectorTensorTest<TNumber>(TNumber[] left, TNumber[,,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftVector = Tensor.FromArray<TNumber>(left).ToVector();
        using var rightTensor = Tensor.FromArray<TNumber>(right).ToTensor();

        leftVector.To(device);
        rightTensor.To(device);

        using var result = leftVector.Multiply(rightTensor);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndScalar(IDevice device)
    {
        MatrixScalarTest<BFloat16>(new BFloat16[,] { { 1, 2 }, { 3, 4 } }, 2, device).Should().BeEquivalentTo(new BFloat16[,] { { 2, 4 }, { 6, 8 } });
        MatrixScalarTest<float>(new float[,] { { 1, 2 }, { 3, 4 } }, 2, device).Should().BeEquivalentTo(new float[,] { { 2, 4 }, { 6, 8 } });
        MatrixScalarTest<double>(new double[,] { { 1, 2 }, { 3, 4 } }, 2, device).Should().BeEquivalentTo(new double[,] { { 2, 4 }, { 6, 8 } });
        MatrixScalarTest<byte>(new byte[,] { { 1, 2 }, { 3, 4 } }, 2, device).Should().BeEquivalentTo(new byte[,] { { 2, 4 }, { 6, 8 } });
        MatrixScalarTest<sbyte>(new sbyte[,] { { 1, 2 }, { 3, 4 } }, 2, device).Should().BeEquivalentTo(new sbyte[,] { { 2, 4 }, { 6, 8 } });
        MatrixScalarTest<ushort>(new ushort[,] { { 1, 2 }, { 3, 4 } }, 2, device).Should().BeEquivalentTo(new ushort[,] { { 2, 4 }, { 6, 8 } });
        MatrixScalarTest<short>(new short[,] { { 1, 2 }, { 3, 4 } }, 2, device).Should().BeEquivalentTo(new short[,] { { 2, 4 }, { 6, 8 } });
        MatrixScalarTest<uint>(new uint[,] { { 1, 2 }, { 3, 4 } }, 2, device).Should().BeEquivalentTo(new uint[,] { { 2, 4 }, { 6, 8 } });
        MatrixScalarTest<int>(new int[,] { { 1, 2 }, { 3, 4 } }, 2, device).Should().BeEquivalentTo(new int[,] { { 2, 4 }, { 6, 8 } });
        MatrixScalarTest<ulong>(new ulong[,] { { 1, 2 }, { 3, 4 } }, 2, device).Should().BeEquivalentTo(new ulong[,] { { 2, 4 }, { 6, 8 } });
        MatrixScalarTest<long>(new long[,] { { 1, 2 }, { 3, 4 } }, 2, device).Should().BeEquivalentTo(new long[,] { { 2, 4 }, { 6, 8 } });
    }

    private static Array MatrixScalarTest<TNumber>(TNumber[,] left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftMatrix = Tensor.FromArray<TNumber>(left).ToMatrix();
        using var rightScalar = new Scalar<TNumber>(right);

        leftMatrix.To(device);
        rightScalar.To(device);

        using var result = leftMatrix.Multiply(rightScalar);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndVector(IDevice device)
    {
        MatrixVectorTest<BFloat16>(new BFloat16[,] { { 1, 2 }, { 3, 4 } }, new BFloat16[] { 1, 2 }, device).Should().BeEquivalentTo(new BFloat16[,] { { 1, 4 }, { 3, 8 } });
        MatrixVectorTest<float>(new float[,] { { 1, 2 }, { 3, 4 } }, new float[] { 1, 2 }, device).Should().BeEquivalentTo(new float[,] { { 1, 4 }, { 3, 8 } });
        MatrixVectorTest<double>(new double[,] { { 1, 2 }, { 3, 4 } }, new double[] { 1, 2 }, device).Should().BeEquivalentTo(new double[,] { { 1, 4 }, { 3, 8 } });
        MatrixVectorTest<byte>(new byte[,] { { 1, 2 }, { 3, 4 } }, new byte[] { 1, 2 }, device).Should().BeEquivalentTo(new byte[,] { { 1, 4 }, { 3, 8 } });
        MatrixVectorTest<sbyte>(new sbyte[,] { { 1, 2 }, { 3, 4 } }, new sbyte[] { 1, 2 }, device).Should().BeEquivalentTo(new sbyte[,] { { 1, 4 }, { 3, 8 } });
        MatrixVectorTest<ushort>(new ushort[,] { { 1, 2 }, { 3, 4 } }, new ushort[] { 1, 2 }, device).Should().BeEquivalentTo(new ushort[,] { { 1, 4 }, { 3, 8 } });
        MatrixVectorTest<short>(new short[,] { { 1, 2 }, { 3, 4 } }, new short[] { 1, 2 }, device).Should().BeEquivalentTo(new short[,] { { 1, 4 }, { 3, 8 } });
        MatrixVectorTest<uint>(new uint[,] { { 1, 2 }, { 3, 4 } }, new uint[] { 1, 2 }, device).Should().BeEquivalentTo(new uint[,] { { 1, 4 }, { 3, 8 } });
        MatrixVectorTest<int>(new int[,] { { 1, 2 }, { 3, 4 } }, new int[] { 1, 2 }, device).Should().BeEquivalentTo(new int[,] { { 1, 4 }, { 3, 8 } });
        MatrixVectorTest<ulong>(new ulong[,] { { 1, 2 }, { 3, 4 } }, new ulong[] { 1, 2 }, device).Should().BeEquivalentTo(new ulong[,] { { 1, 4 }, { 3, 8 } });
        MatrixVectorTest<long>(new long[,] { { 1, 2 }, { 3, 4 } }, new long[] { 1, 2 }, device).Should().BeEquivalentTo(new long[,] { { 1, 4 }, { 3, 8 } });
    }

    private static Array MatrixVectorTest<TNumber>(TNumber[,] left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftMatrix = Tensor.FromArray<TNumber>(left).ToMatrix();
        using var rightVector = Tensor.FromArray<TNumber>(right).ToVector();

        leftMatrix.To(device);
        rightVector.To(device);

        using var result = leftMatrix.Multiply(rightVector);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndMatrix(IDevice device)
    {
        MatrixMatrixTest<BFloat16>(new BFloat16[,] { { 1, 2 }, { 3, 4 } }, new BFloat16[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new BFloat16[,] { { 1, 4 }, { 9, 16 } });
        MatrixMatrixTest<float>(new float[,] { { 1, 2 }, { 3, 4 } }, new float[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new float[,] { { 1, 4 }, { 9, 16 } });
        MatrixMatrixTest<double>(new double[,] { { 1, 2 }, { 3, 4 } }, new double[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new double[,] { { 1, 4 }, { 9, 16 } });
        MatrixMatrixTest<byte>(new byte[,] { { 1, 2 }, { 3, 4 } }, new byte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new byte[,] { { 1, 4 }, { 9, 16 } });
        MatrixMatrixTest<sbyte>(new sbyte[,] { { 1, 2 }, { 3, 4 } }, new sbyte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new sbyte[,] { { 1, 4 }, { 9, 16 } });
        MatrixMatrixTest<ushort>(new ushort[,] { { 1, 2 }, { 3, 4 } }, new ushort[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ushort[,] { { 1, 4 }, { 9, 16 } });
        MatrixMatrixTest<short>(new short[,] { { 1, 2 }, { 3, 4 } }, new short[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new short[,] { { 1, 4 }, { 9, 16 } });
        MatrixMatrixTest<uint>(new uint[,] { { 1, 2 }, { 3, 4 } }, new uint[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new uint[,] { { 1, 4 }, { 9, 16 } });
        MatrixMatrixTest<int>(new int[,] { { 1, 2 }, { 3, 4 } }, new int[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new int[,] { { 1, 4 }, { 9, 16 } });
        MatrixMatrixTest<ulong>(new ulong[,] { { 1, 2 }, { 3, 4 } }, new ulong[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ulong[,] { { 1, 4 }, { 9, 16 } });
        MatrixMatrixTest<long>(new long[,] { { 1, 2 }, { 3, 4 } }, new long[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new long[,] { { 1, 4 }, { 9, 16 } });
    }

    private static Array MatrixMatrixTest<TNumber>(TNumber[,] left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftMatrix = Tensor.FromArray<TNumber>(left).ToMatrix();
        using var rightMatrix = Tensor.FromArray<TNumber>(right).ToMatrix();

        leftMatrix.To(device);
        rightMatrix.To(device);

        using var result = leftMatrix.Multiply(rightMatrix);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndTensor(IDevice device)
    {
        MatrixTensorTest<BFloat16>(new BFloat16[,] { { 1, 2 }, { 3, 4 } }, new BFloat16[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new BFloat16[,,] { { { 1, 4 }, { 9, 16 } }, { { 5, 12 }, { 21, 32 } } });
        MatrixTensorTest<float>(new float[,] { { 1, 2 }, { 3, 4 } }, new float[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new float[,,] { { { 1, 4 }, { 9, 16 } }, { { 5, 12 }, { 21, 32 } } });
        MatrixTensorTest<double>(new double[,] { { 1, 2 }, { 3, 4 } }, new double[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new double[,,] { { { 1, 4 }, { 9, 16 } }, { { 5, 12 }, { 21, 32 } } });
        MatrixTensorTest<byte>(new byte[,] { { 1, 2 }, { 3, 4 } }, new byte[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new byte[,,] { { { 1, 4 }, { 9, 16 } }, { { 5, 12 }, { 21, 32 } } });
        MatrixTensorTest<sbyte>(new sbyte[,] { { 1, 2 }, { 3, 4 } }, new sbyte[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new sbyte[,,] { { { 1, 4 }, { 9, 16 } }, { { 5, 12 }, { 21, 32 } } });
        MatrixTensorTest<ushort>(new ushort[,] { { 1, 2 }, { 3, 4 } }, new ushort[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new ushort[,,] { { { 1, 4 }, { 9, 16 } }, { { 5, 12 }, { 21, 32 } } });
        MatrixTensorTest<short>(new short[,] { { 1, 2 }, { 3, 4 } }, new short[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new short[,,] { { { 1, 4 }, { 9, 16 } }, { { 5, 12 }, { 21, 32 } } });
        MatrixTensorTest<uint>(new uint[,] { { 1, 2 }, { 3, 4 } }, new uint[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new uint[,,] { { { 1, 4 }, { 9, 16 } }, { { 5, 12 }, { 21, 32 } } });
        MatrixTensorTest<int>(new int[,] { { 1, 2 }, { 3, 4 } }, new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new int[,,] { { { 1, 4 }, { 9, 16 } }, { { 5, 12 }, { 21, 32 } } });
        MatrixTensorTest<ulong>(new ulong[,] { { 1, 2 }, { 3, 4 } }, new ulong[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new ulong[,,] { { { 1, 4 }, { 9, 16 } }, { { 5, 12 }, { 21, 32 } } });
        MatrixTensorTest<long>(new long[,] { { 1, 2 }, { 3, 4 } }, new long[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device).Should().BeEquivalentTo(new long[,,] { { { 1, 4 }, { 9, 16 } }, { { 5, 12 }, { 21, 32 } } });
    }

    private static Array MatrixTensorTest<TNumber>(TNumber[,] left, TNumber[,,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftMatrix = Tensor.FromArray<TNumber>(left).ToMatrix();
        using var rightTensor = Tensor.FromArray<TNumber>(right).ToTensor();

        leftMatrix.To(device);
        rightTensor.To(device);

        using var result = leftMatrix.Multiply(rightTensor);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorAndScalar(IDevice device)
    {
        TensorScalarTest<BFloat16>(new BFloat16[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, 2, device).Should().BeEquivalentTo(new BFloat16[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        TensorScalarTest<float>(new float[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, 2, device).Should().BeEquivalentTo(new float[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        TensorScalarTest<double>(new double[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, 2, device).Should().BeEquivalentTo(new double[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        TensorScalarTest<byte>(new byte[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, 2, device).Should().BeEquivalentTo(new byte[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        TensorScalarTest<sbyte>(new sbyte[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, 2, device).Should().BeEquivalentTo(new sbyte[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        TensorScalarTest<ushort>(new ushort[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, 2, device).Should().BeEquivalentTo(new ushort[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        TensorScalarTest<short>(new short[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, 2, device).Should().BeEquivalentTo(new short[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        TensorScalarTest<uint>(new uint[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, 2, device).Should().BeEquivalentTo(new uint[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        TensorScalarTest<int>(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, 2, device).Should().BeEquivalentTo(new int[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        TensorScalarTest<ulong>(new ulong[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, 2, device).Should().BeEquivalentTo(new ulong[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
        TensorScalarTest<long>(new long[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, 2, device).Should().BeEquivalentTo(new long[,,] { { { 2, 4 }, { 6, 8 } }, { { 10, 12 }, { 14, 16 } } });
    }

    private static Array TensorScalarTest<TNumber>(TNumber[,,] left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftTensor = Tensor.FromArray<TNumber>(left).ToTensor();
        using var rightScalar = new Scalar<TNumber>(right);

        leftTensor.To(device);
        rightScalar.To(device);

        using var result = leftTensor.Multiply(rightScalar);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorAndVector(IDevice device)
    {
        TensorVectorTest<BFloat16>(new BFloat16[,,] { { { 1, 2 }, { 3, 4 } } }, new BFloat16[] { 1, 2 }, device).Should().BeEquivalentTo(new BFloat16[,,] { { { 1, 4 }, { 3, 8 } } });
        TensorVectorTest<float>(new float[,,] { { { 1, 2 }, { 3, 4 } } }, new float[] { 1, 2 }, device).Should().BeEquivalentTo(new float[,,] { { { 1, 4 }, { 3, 8 } } });
        TensorVectorTest<double>(new double[,,] { { { 1, 2 }, { 3, 4 } } }, new double[] { 1, 2 }, device).Should().BeEquivalentTo(new double[,,] { { { 1, 4 }, { 3, 8 } } });
        TensorVectorTest<byte>(new byte[,,] { { { 1, 2 }, { 3, 4 } } }, new byte[] { 1, 2 }, device).Should().BeEquivalentTo(new byte[,,] { { { 1, 4 }, { 3, 8 } } });
        TensorVectorTest<sbyte>(new sbyte[,,] { { { 1, 2 }, { 3, 4 } } }, new sbyte[] { 1, 2 }, device).Should().BeEquivalentTo(new sbyte[,,] { { { 1, 4 }, { 3, 8 } } });
        TensorVectorTest<ushort>(new ushort[,,] { { { 1, 2 }, { 3, 4 } } }, new ushort[] { 1, 2 }, device).Should().BeEquivalentTo(new ushort[,,] { { { 1, 4 }, { 3, 8 } } });
        TensorVectorTest<short>(new short[,,] { { { 1, 2 }, { 3, 4 } } }, new short[] { 1, 2 }, device).Should().BeEquivalentTo(new short[,,] { { { 1, 4 }, { 3, 8 } } });
        TensorVectorTest<uint>(new uint[,,] { { { 1, 2 }, { 3, 4 } } }, new uint[] { 1, 2 }, device).Should().BeEquivalentTo(new uint[,,] { { { 1, 4 }, { 3, 8 } } });
        TensorVectorTest<int>(new int[,,] { { { 1, 2 }, { 3, 4 } } }, new int[] { 1, 2 }, device).Should().BeEquivalentTo(new int[,,] { { { 1, 4 }, { 3, 8 } } });
        TensorVectorTest<ulong>(new ulong[,,] { { { 1, 2 }, { 3, 4 } } }, new ulong[] { 1, 2 }, device).Should().BeEquivalentTo(new ulong[,,] { { { 1, 4 }, { 3, 8 } } });
        TensorVectorTest<long>(new long[,,] { { { 1, 2 }, { 3, 4 } } }, new long[] { 1, 2 }, device).Should().BeEquivalentTo(new long[,,] { { { 1, 4 }, { 3, 8 } } });
    }

    private static Array TensorVectorTest<TNumber>(TNumber[,,] left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftTensor = Tensor.FromArray<TNumber>(left).ToTensor();
        using var rightVector = Tensor.FromArray<TNumber>(right).ToVector();

        leftTensor.To(device);
        rightVector.To(device);

        using var result = leftTensor.Multiply(rightVector);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorAndMatrix(IDevice device)
    {
        TensorMatrixTest<BFloat16>(new BFloat16[,,] { { { 1, 2 }, { 3, 4 } } }, new BFloat16[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new BFloat16[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorMatrixTest<float>(new float[,,] { { { 1, 2 }, { 3, 4 } } }, new float[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new float[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorMatrixTest<double>(new double[,,] { { { 1, 2 }, { 3, 4 } } }, new double[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new double[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorMatrixTest<byte>(new byte[,,] { { { 1, 2 }, { 3, 4 } } }, new byte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new byte[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorMatrixTest<sbyte>(new sbyte[,,] { { { 1, 2 }, { 3, 4 } } }, new sbyte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new sbyte[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorMatrixTest<ushort>(new ushort[,,] { { { 1, 2 }, { 3, 4 } } }, new ushort[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ushort[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorMatrixTest<short>(new short[,,] { { { 1, 2 }, { 3, 4 } } }, new short[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new short[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorMatrixTest<uint>(new uint[,,] { { { 1, 2 }, { 3, 4 } } }, new uint[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new uint[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorMatrixTest<int>(new int[,,] { { { 1, 2 }, { 3, 4 } } }, new int[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new int[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorMatrixTest<ulong>(new ulong[,,] { { { 1, 2 }, { 3, 4 } } }, new ulong[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ulong[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorMatrixTest<long>(new long[,,] { { { 1, 2 }, { 3, 4 } } }, new long[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new long[,,] { { { 1, 4 }, { 9, 16 } } });
    }

    private static Array TensorMatrixTest<TNumber>(TNumber[,,] left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftTensor = Tensor.FromArray<TNumber>(left).ToTensor();
        using var rightMatrix = Tensor.FromArray<TNumber>(right).ToMatrix();

        leftTensor.To(device);
        rightMatrix.To(device);

        using var result = leftTensor.Multiply(rightMatrix);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorAndTensor(IDevice device)
    {
        TensorTensorTest<BFloat16>(new BFloat16[,,] { { { 1, 2 }, { 3, 4 } } }, new BFloat16[,,] { { { 1, 2 }, { 3, 4 } } }, device).Should().BeEquivalentTo(new BFloat16[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorTensorTest<float>(new float[,,] { { { 1, 2 }, { 3, 4 } } }, new float[,,] { { { 1, 2 }, { 3, 4 } } }, device).Should().BeEquivalentTo(new float[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorTensorTest<double>(new double[,,] { { { 1, 2 }, { 3, 4 } } }, new double[,,] { { { 1, 2 }, { 3, 4 } } }, device).Should().BeEquivalentTo(new double[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorTensorTest<byte>(new byte[,,] { { { 1, 2 }, { 3, 4 } } }, new byte[,,] { { { 1, 2 }, { 3, 4 } } }, device).Should().BeEquivalentTo(new byte[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorTensorTest<sbyte>(new sbyte[,,] { { { 1, 2 }, { 3, 4 } } }, new sbyte[,,] { { { 1, 2 }, { 3, 4 } } }, device).Should().BeEquivalentTo(new sbyte[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorTensorTest<ushort>(new ushort[,,] { { { 1, 2 }, { 3, 4 } } }, new ushort[,,] { { { 1, 2 }, { 3, 4 } } }, device).Should().BeEquivalentTo(new ushort[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorTensorTest<short>(new short[,,] { { { 1, 2 }, { 3, 4 } } }, new short[,,] { { { 1, 2 }, { 3, 4 } } }, device).Should().BeEquivalentTo(new short[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorTensorTest<uint>(new uint[,,] { { { 1, 2 }, { 3, 4 } } }, new uint[,,] { { { 1, 2 }, { 3, 4 } } }, device).Should().BeEquivalentTo(new uint[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorTensorTest<int>(new int[,,] { { { 1, 2 }, { 3, 4 } } }, new int[,,] { { { 1, 2 }, { 3, 4 } } }, device).Should().BeEquivalentTo(new int[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorTensorTest<ulong>(new ulong[,,] { { { 1, 2 }, { 3, 4 } } }, new ulong[,,] { { { 1, 2 }, { 3, 4 } } }, device).Should().BeEquivalentTo(new ulong[,,] { { { 1, 4 }, { 9, 16 } } });
        TensorTensorTest<long>(new long[,,] { { { 1, 2 }, { 3, 4 } } }, new long[,,] { { { 1, 2 }, { 3, 4 } } }, device).Should().BeEquivalentTo(new long[,,] { { { 1, 4 }, { 9, 16 } } });
    }

    private static Array TensorTensorTest<TNumber>(TNumber[,,] left, TNumber[,,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftTensor = Tensor.FromArray<TNumber>(left).ToTensor();
        using var rightTensor = Tensor.FromArray<TNumber>(right).ToTensor();

        leftTensor.To(device);
        rightTensor.To(device);

        using var result = leftTensor.Multiply(rightTensor);

        return result.ToArray();
    }
}