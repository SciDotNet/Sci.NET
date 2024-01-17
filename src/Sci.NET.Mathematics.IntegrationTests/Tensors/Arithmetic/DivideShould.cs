// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Arithmetic;

public class DivideShould : IntegrationTestBase, IArithmeticTests
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarsAndScalar(IDevice device)
    {
        ScalarScalarTest<BFloat16>(4, 2, device).Should().Be(2);
        ScalarScalarTest<float>(4, 2, device).Should().Be(2);
        ScalarScalarTest<double>(4, 2, device).Should().Be(2);
        ScalarScalarTest<decimal>(4, 2, device).Should().Be(2);
        ScalarScalarTest<byte>(4, 2, device).Should().Be(2);
        ScalarScalarTest<sbyte>(4, 2, device).Should().Be(2);
        ScalarScalarTest<ushort>(4, 2, device).Should().Be(2);
        ScalarScalarTest<short>(4, 2, device).Should().Be(2);
        ScalarScalarTest<uint>(4, 2, device).Should().Be(2);
        ScalarScalarTest<int>(4, 2, device).Should().Be(2);
        ScalarScalarTest<ulong>(4, 2, device).Should().Be(2);
        ScalarScalarTest<long>(4, 2, device).Should().Be(2);
    }

    private static TNumber ScalarScalarTest<TNumber>(TNumber left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftScalar = new Scalar<TNumber>(left);
        using var rightScalar = new Scalar<TNumber>(right);

        leftScalar.To(device);
        rightScalar.To(device);

        using var result = leftScalar.Divide(rightScalar);

        return result.Value;
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarAndVector(IDevice device)
    {
        ScalarVectorTest<BFloat16>(16, new BFloat16[] { 2, 4, 8, 16 }, device).Should().BeEquivalentTo(new BFloat16[] { 8, 4, 2, 1 });
        ScalarVectorTest<float>(16, new float[] { 2, 4, 8, 16 }, device).Should().BeEquivalentTo(new float[] { 8, 4, 2, 1 });
        ScalarVectorTest<double>(16, new double[] { 2, 4, 8, 16 }, device).Should().BeEquivalentTo(new double[] { 8, 4, 2, 1 });
        ScalarVectorTest<byte>(16, new byte[] { 2, 4, 8, 16 }, device).Should().BeEquivalentTo(new byte[] { 8, 4, 2, 1 });
        ScalarVectorTest<sbyte>(16, new sbyte[] { 2, 4, 8, 16 }, device).Should().BeEquivalentTo(new sbyte[] { 8, 4, 2, 1 });
        ScalarVectorTest<ushort>(16, new ushort[] { 2, 4, 8, 16 }, device).Should().BeEquivalentTo(new ushort[] { 8, 4, 2, 1 });
        ScalarVectorTest<short>(16, new short[] { 2, 4, 8, 16 }, device).Should().BeEquivalentTo(new short[] { 8, 4, 2, 1 });
        ScalarVectorTest<uint>(16, new uint[] { 2, 4, 8, 16 }, device).Should().BeEquivalentTo(new uint[] { 8, 4, 2, 1 });
        ScalarVectorTest<int>(16, new int[] { 2, 4, 8, 16 }, device).Should().BeEquivalentTo(new int[] { 8, 4, 2, 1 });
        ScalarVectorTest<ulong>(16, new ulong[] { 2, 4, 8, 16 }, device).Should().BeEquivalentTo(new ulong[] { 8, 4, 2, 1 });
        ScalarVectorTest<long>(16, new long[] { 2, 4, 8, 16 }, device).Should().BeEquivalentTo(new long[] { 8, 4, 2, 1 });
    }

    private static Array ScalarVectorTest<TNumber>(TNumber left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftVector = new Scalar<TNumber>(left);
        using var rightScalar = Tensor.FromArray<TNumber>(right).ToVector();

        leftVector.To(device);
        rightScalar.To(device);

        using var result = leftVector.Divide(rightScalar);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarAndMatrix(IDevice device)
    {
        ScalarMatrixTest<BFloat16>(32, new BFloat16[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new BFloat16[,] { { 16, 8 }, { 4, 2 } });
        ScalarMatrixTest<float>(32, new float[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new float[,] { { 16, 8 }, { 4, 2 } });
        ScalarMatrixTest<double>(32, new double[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new double[,] { { 16, 8 }, { 4, 2 } });
        ScalarMatrixTest<byte>(32, new byte[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new byte[,] { { 16, 8 }, { 4, 2 } });
        ScalarMatrixTest<sbyte>(32, new sbyte[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new sbyte[,] { { 16, 8 }, { 4, 2 } });
        ScalarMatrixTest<ushort>(32, new ushort[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new ushort[,] { { 16, 8 }, { 4, 2 } });
        ScalarMatrixTest<short>(32, new short[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new short[,] { { 16, 8 }, { 4, 2 } });
        ScalarMatrixTest<uint>(32, new uint[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new uint[,] { { 16, 8 }, { 4, 2 } });
        ScalarMatrixTest<int>(32, new int[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new int[,] { { 16, 8 }, { 4, 2 } });
        ScalarMatrixTest<ulong>(32, new ulong[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new ulong[,] { { 16, 8 }, { 4, 2 } });
        ScalarMatrixTest<long>(32, new long[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new long[,] { { 16, 8 }, { 4, 2 } });
    }

    private static Array ScalarMatrixTest<TNumber>(TNumber left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftVector = new Scalar<TNumber>(left);
        using var rightScalar = Tensor.FromArray<TNumber>(right).ToMatrix();

        leftVector.To(device);
        rightScalar.To(device);

        using var result = leftVector.Divide(rightScalar);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarTensor(IDevice device)
    {
        ScalarTensorTest<BFloat16>(64, new BFloat16[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new BFloat16[,,] { { { 32, 16 }, { 8, 4 } }, { { 32, 16 }, { 8, 4 } } });
        ScalarTensorTest<float>(64, new float[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new float[,,] { { { 32, 16 }, { 8, 4 } }, { { 32, 16 }, { 8, 4 } } });
        ScalarTensorTest<double>(64, new double[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new double[,,] { { { 32, 16 }, { 8, 4 } }, { { 32, 16 }, { 8, 4 } } });
        ScalarTensorTest<byte>(64, new byte[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new byte[,,] { { { 32, 16 }, { 8, 4 } }, { { 32, 16 }, { 8, 4 } } });
        ScalarTensorTest<sbyte>(64, new sbyte[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new sbyte[,,] { { { 32, 16 }, { 8, 4 } }, { { 32, 16 }, { 8, 4 } } });
        ScalarTensorTest<ushort>(64, new ushort[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new ushort[,,] { { { 32, 16 }, { 8, 4 } }, { { 32, 16 }, { 8, 4 } } });
        ScalarTensorTest<short>(64, new short[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new short[,,] { { { 32, 16 }, { 8, 4 } }, { { 32, 16 }, { 8, 4 } } });
        ScalarTensorTest<uint>(64, new uint[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new uint[,,] { { { 32, 16 }, { 8, 4 } }, { { 32, 16 }, { 8, 4 } } });
        ScalarTensorTest<int>(64, new int[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new int[,,] { { { 32, 16 }, { 8, 4 } }, { { 32, 16 }, { 8, 4 } } });
        ScalarTensorTest<ulong>(64, new ulong[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new ulong[,,] { { { 32, 16 }, { 8, 4 } }, { { 32, 16 }, { 8, 4 } } });
        ScalarTensorTest<long>(64, new long[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new long[,,] { { { 32, 16 }, { 8, 4 } }, { { 32, 16 }, { 8, 4 } } });
    }

    private static Array ScalarTensorTest<TNumber>(TNumber left, TNumber[,,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftVector = new Scalar<TNumber>(left);
        using var rightScalar = Tensor.FromArray<TNumber>(right).ToTensor();

        leftVector.To(device);
        rightScalar.To(device);

        using var result = leftVector.Divide(rightScalar);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndScalar(IDevice device)
    {
        VectorScalarTest<BFloat16>(new BFloat16[] { 2, 4, 8, 16 }, 2, device).Should().BeEquivalentTo(new BFloat16[] { 1, 2, 4, 8 });
        VectorScalarTest<float>(new float[] { 2, 4, 8, 16 }, 2, device).Should().BeEquivalentTo(new float[] { 1, 2, 4, 8 });
        VectorScalarTest<double>(new double[] { 2, 4, 8, 16 }, 2, device).Should().BeEquivalentTo(new double[] { 1, 2, 4, 8 });
        VectorScalarTest<byte>(new byte[] { 2, 4, 8, 16 }, 2, device).Should().BeEquivalentTo(new byte[] { 1, 2, 4, 8 });
        VectorScalarTest<sbyte>(new sbyte[] { 2, 4, 8, 16 }, 2, device).Should().BeEquivalentTo(new sbyte[] { 1, 2, 4, 8 });
        VectorScalarTest<ushort>(new ushort[] { 2, 4, 8, 16 }, 2, device).Should().BeEquivalentTo(new ushort[] { 1, 2, 4, 8 });
        VectorScalarTest<short>(new short[] { 2, 4, 8, 16 }, 2, device).Should().BeEquivalentTo(new short[] { 1, 2, 4, 8 });
        VectorScalarTest<uint>(new uint[] { 2, 4, 8, 16 }, 2, device).Should().BeEquivalentTo(new uint[] { 1, 2, 4, 8 });
        VectorScalarTest<int>(new int[] { 2, 4, 8, 16 }, 2, device).Should().BeEquivalentTo(new int[] { 1, 2, 4, 8 });
        VectorScalarTest<ulong>(new ulong[] { 2, 4, 8, 16 }, 2, device).Should().BeEquivalentTo(new ulong[] { 1, 2, 4, 8 });
        VectorScalarTest<long>(new long[] { 2, 4, 8, 16 }, 2, device).Should().BeEquivalentTo(new long[] { 1, 2, 4, 8 });
    }

    private static Array VectorScalarTest<TNumber>(TNumber[] left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftVector = Tensor.FromArray<TNumber>(left).ToVector();
        using var rightScalar = new Scalar<TNumber>(right);

        leftVector.To(device);
        rightScalar.To(device);

        using var result = leftVector.Divide(rightScalar);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndVector(IDevice device)
    {
        VectorVectorTest<BFloat16>(new BFloat16[] { 4, 8, 16, 32 }, new BFloat16[] { 2, 4, 2, 4 }, device).Should().BeEquivalentTo(new BFloat16[] { 2, 2, 8, 8 });
        VectorVectorTest<float>(new float[] { 4, 8, 16, 32 }, new float[] { 2, 4, 2, 4 }, device).Should().BeEquivalentTo(new float[] { 2, 2, 8, 8 });
        VectorVectorTest<double>(new double[] { 4, 8, 16, 32 }, new double[] { 2, 4, 2, 4 }, device).Should().BeEquivalentTo(new double[] { 2, 2, 8, 8 });
        VectorVectorTest<byte>(new byte[] { 4, 8, 16, 32 }, new byte[] { 2, 4, 2, 4 }, device).Should().BeEquivalentTo(new byte[] { 2, 2, 8, 8 });
        VectorVectorTest<sbyte>(new sbyte[] { 4, 8, 16, 32 }, new sbyte[] { 2, 4, 2, 4 }, device).Should().BeEquivalentTo(new sbyte[] { 2, 2, 8, 8 });
        VectorVectorTest<ushort>(new ushort[] { 4, 8, 16, 32 }, new ushort[] { 2, 4, 2, 4 }, device).Should().BeEquivalentTo(new ushort[] { 2, 2, 8, 8 });
        VectorVectorTest<short>(new short[] { 4, 8, 16, 32 }, new short[] { 2, 4, 2, 4 }, device).Should().BeEquivalentTo(new short[] { 2, 2, 8, 8 });
        VectorVectorTest<uint>(new uint[] { 4, 8, 16, 32 }, new uint[] { 2, 4, 2, 4 }, device).Should().BeEquivalentTo(new uint[] { 2, 2, 8, 8 });
        VectorVectorTest<int>(new int[] { 4, 8, 16, 32 }, new int[] { 2, 4, 2, 4 }, device).Should().BeEquivalentTo(new int[] { 2, 2, 8, 8 });
        VectorVectorTest<ulong>(new ulong[] { 4, 8, 16, 32 }, new ulong[] { 2, 4, 2, 4 }, device).Should().BeEquivalentTo(new ulong[] { 2, 2, 8, 8 });
        VectorVectorTest<long>(new long[] { 4, 8, 16, 32 }, new long[] { 2, 4, 2, 4 }, device).Should().BeEquivalentTo(new long[] { 2, 2, 8, 8 });
    }

    private static Array VectorVectorTest<TNumber>(TNumber[] left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftVector = Tensor.FromArray<TNumber>(left).ToVector();
        using var rightVector = Tensor.FromArray<TNumber>(right).ToVector();

        leftVector.To(device);
        rightVector.To(device);

        using var result = leftVector.Divide(rightVector);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndMatrix(IDevice device)
    {
        VectorMatrixTest<BFloat16>(new BFloat16[] { 4, 8, 16, 32 }, new BFloat16[,] { { 2, 4, 8, 16 }, { 2, 4, 8, 16 } }, device).Should().BeEquivalentTo(new BFloat16[,] { { 2, 2, 2, 2 }, { 2, 2, 2, 2 } });
        VectorMatrixTest<float>(new float[] { 4, 8, 16, 32 }, new float[,] { { 2, 4, 8, 16 }, { 2, 4, 8, 16 } }, device).Should().BeEquivalentTo(new float[,] { { 2, 2, 2, 2 }, { 2, 2, 2, 2 } });
        VectorMatrixTest<double>(new double[] { 4, 8, 16, 32 }, new double[,] { { 2, 4, 8, 16 }, { 2, 4, 8, 16 } }, device).Should().BeEquivalentTo(new double[,] { { 2, 2, 2, 2 }, { 2, 2, 2, 2 } });
        VectorMatrixTest<byte>(new byte[] { 4, 8, 16, 32 }, new byte[,] { { 2, 4, 8, 16 }, { 2, 4, 8, 16 } }, device).Should().BeEquivalentTo(new byte[,] { { 2, 2, 2, 2 }, { 2, 2, 2, 2 } });
        VectorMatrixTest<sbyte>(new sbyte[] { 4, 8, 16, 32 }, new sbyte[,] { { 2, 4, 8, 16 }, { 2, 4, 8, 16 } }, device).Should().BeEquivalentTo(new sbyte[,] { { 2, 2, 2, 2 }, { 2, 2, 2, 2 } });
        VectorMatrixTest<ushort>(new ushort[] { 4, 8, 16, 32 }, new ushort[,] { { 2, 4, 8, 16 }, { 2, 4, 8, 16 } }, device).Should().BeEquivalentTo(new ushort[,] { { 2, 2, 2, 2 }, { 2, 2, 2, 2 } });
        VectorMatrixTest<short>(new short[] { 4, 8, 16, 32 }, new short[,] { { 2, 4, 8, 16 }, { 2, 4, 8, 16 } }, device).Should().BeEquivalentTo(new short[,] { { 2, 2, 2, 2 }, { 2, 2, 2, 2 } });
        VectorMatrixTest<uint>(new uint[] { 4, 8, 16, 32 }, new uint[,] { { 2, 4, 8, 16 }, { 2, 4, 8, 16 } }, device).Should().BeEquivalentTo(new uint[,] { { 2, 2, 2, 2 }, { 2, 2, 2, 2 } });
        VectorMatrixTest<int>(new int[] { 4, 8, 16, 32 }, new int[,] { { 2, 4, 8, 16 }, { 2, 4, 8, 16 } }, device).Should().BeEquivalentTo(new int[,] { { 2, 2, 2, 2 }, { 2, 2, 2, 2 } });
        VectorMatrixTest<ulong>(new ulong[] { 4, 8, 16, 32 }, new ulong[,] { { 2, 4, 8, 16 }, { 2, 4, 8, 16 } }, device).Should().BeEquivalentTo(new ulong[,] { { 2, 2, 2, 2 }, { 2, 2, 2, 2 } });
        VectorMatrixTest<long>(new long[] { 4, 8, 16, 32 }, new long[,] { { 2, 4, 8, 16 }, { 2, 4, 8, 16 } }, device).Should().BeEquivalentTo(new long[,] { { 2, 2, 2, 2 }, { 2, 2, 2, 2 } });
    }

    private static Array VectorMatrixTest<TNumber>(TNumber[] left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftVector = Tensor.FromArray<TNumber>(left).ToVector();
        using var rightMatrix = Tensor.FromArray<TNumber>(right).ToMatrix();

        leftVector.To(device);
        rightMatrix.To(device);

        using var result = leftVector.Divide(rightMatrix);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndTensor(IDevice device)
    {
        VectorTensorTest<BFloat16>(new BFloat16[] { 16, 32 }, new BFloat16[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new BFloat16[,,] { { { 8, 8 }, { 2, 2 } }, { { 8, 8 }, { 2, 2 } } });
        VectorTensorTest<float>(new float[] { 16, 32 }, new float[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new float[,,] { { { 8, 8 }, { 2, 2 } }, { { 8, 8 }, { 2, 2 } } });
        VectorTensorTest<double>(new double[] { 16, 32 }, new double[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new double[,,] { { { 8, 8 }, { 2, 2 } }, { { 8, 8 }, { 2, 2 } } });
        VectorTensorTest<byte>(new byte[] { 16, 32 }, new byte[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new byte[,,] { { { 8, 8 }, { 2, 2 } }, { { 8, 8 }, { 2, 2 } } });
        VectorTensorTest<sbyte>(new sbyte[] { 16, 32 }, new sbyte[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new sbyte[,,] { { { 8, 8 }, { 2, 2 } }, { { 8, 8 }, { 2, 2 } } });
        VectorTensorTest<ushort>(new ushort[] { 16, 32 }, new ushort[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new ushort[,,] { { { 8, 8 }, { 2, 2 } }, { { 8, 8 }, { 2, 2 } } });
        VectorTensorTest<short>(new short[] { 16, 32 }, new short[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new short[,,] { { { 8, 8 }, { 2, 2 } }, { { 8, 8 }, { 2, 2 } } });
        VectorTensorTest<uint>(new uint[] { 16, 32 }, new uint[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new uint[,,] { { { 8, 8 }, { 2, 2 } }, { { 8, 8 }, { 2, 2 } } });
        VectorTensorTest<int>(new int[] { 16, 32 }, new int[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new int[,,] { { { 8, 8 }, { 2, 2 } }, { { 8, 8 }, { 2, 2 } } });
        VectorTensorTest<ulong>(new ulong[] { 16, 32 }, new ulong[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new ulong[,,] { { { 8, 8 }, { 2, 2 } }, { { 8, 8 }, { 2, 2 } } });
        VectorTensorTest<long>(new long[] { 16, 32 }, new long[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new long[,,] { { { 8, 8 }, { 2, 2 } }, { { 8, 8 }, { 2, 2 } } });
    }

    private static Array VectorTensorTest<TNumber>(TNumber[] left, TNumber[,,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftVector = Tensor.FromArray<TNumber>(left).ToVector();
        using var rightTensor = Tensor.FromArray<TNumber>(right).ToTensor();

        leftVector.To(device);
        rightTensor.To(device);

        using var result = leftVector.Divide(rightTensor);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndScalar(IDevice device)
    {
        MatrixScalarTest<BFloat16>(new BFloat16[,] { { 2, 4 }, { 6, 8 } }, 2, device).Should().BeEquivalentTo(new BFloat16[,] { { 1, 2 }, { 3, 4 } });
        MatrixScalarTest<float>(new float[,] { { 2, 4 }, { 6, 8 } }, 2, device).Should().BeEquivalentTo(new float[,] { { 1, 2 }, { 3, 4 } });
        MatrixScalarTest<double>(new double[,] { { 2, 4 }, { 6, 8 } }, 2, device).Should().BeEquivalentTo(new double[,] { { 1, 2 }, { 3, 4 } });
        MatrixScalarTest<byte>(new byte[,] { { 2, 4 }, { 6, 8 } }, 2, device).Should().BeEquivalentTo(new byte[,] { { 1, 2 }, { 3, 4 } });
        MatrixScalarTest<sbyte>(new sbyte[,] { { 2, 4 }, { 6, 8 } }, 2, device).Should().BeEquivalentTo(new sbyte[,] { { 1, 2 }, { 3, 4 } });
        MatrixScalarTest<ushort>(new ushort[,] { { 2, 4 }, { 6, 8 } }, 2, device).Should().BeEquivalentTo(new ushort[,] { { 1, 2 }, { 3, 4 } });
        MatrixScalarTest<short>(new short[,] { { 2, 4 }, { 6, 8 } }, 2, device).Should().BeEquivalentTo(new short[,] { { 1, 2 }, { 3, 4 } });
        MatrixScalarTest<uint>(new uint[,] { { 2, 4 }, { 6, 8 } }, 2, device).Should().BeEquivalentTo(new uint[,] { { 1, 2 }, { 3, 4 } });
        MatrixScalarTest<int>(new int[,] { { 2, 4 }, { 6, 8 } }, 2, device).Should().BeEquivalentTo(new int[,] { { 1, 2 }, { 3, 4 } });
        MatrixScalarTest<ulong>(new ulong[,] { { 2, 4 }, { 6, 8 } }, 2, device).Should().BeEquivalentTo(new ulong[,] { { 1, 2 }, { 3, 4 } });
        MatrixScalarTest<long>(new long[,] { { 2, 4 }, { 6, 8 } }, 2, device).Should().BeEquivalentTo(new long[,] { { 1, 2 }, { 3, 4 } });
    }

    private static Array MatrixScalarTest<TNumber>(TNumber[,] left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftMatrix = Tensor.FromArray<TNumber>(left).ToMatrix();
        using var rightScalar = new Scalar<TNumber>(right);

        leftMatrix.To(device);
        rightScalar.To(device);

        using var result = leftMatrix.Divide(rightScalar);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndVector(IDevice device)
    {
        MatrixVectorTest<BFloat16>(new BFloat16[,] { { 2, 4 }, { 8, 16 } }, new BFloat16[] { 2, 4 }, device).Should().BeEquivalentTo(new BFloat16[,] { { 1, 1 }, { 4, 4 } });
        MatrixVectorTest<float>(new float[,] { { 2, 4 }, { 8, 16 } }, new float[] { 2, 4 }, device).Should().BeEquivalentTo(new float[,] { { 1, 1 }, { 4, 4 } });
        MatrixVectorTest<double>(new double[,] { { 2, 4 }, { 8, 16 } }, new double[] { 2, 4 }, device).Should().BeEquivalentTo(new double[,] { { 1, 1 }, { 4, 4 } });
        MatrixVectorTest<byte>(new byte[,] { { 2, 4 }, { 8, 16 } }, new byte[] { 2, 4 }, device).Should().BeEquivalentTo(new byte[,] { { 1, 1 }, { 4, 4 } });
        MatrixVectorTest<sbyte>(new sbyte[,] { { 2, 4 }, { 8, 16 } }, new sbyte[] { 2, 4 }, device).Should().BeEquivalentTo(new sbyte[,] { { 1, 1 }, { 4, 4 } });
        MatrixVectorTest<ushort>(new ushort[,] { { 2, 4 }, { 8, 16 } }, new ushort[] { 2, 4 }, device).Should().BeEquivalentTo(new ushort[,] { { 1, 1 }, { 4, 4 } });
        MatrixVectorTest<short>(new short[,] { { 2, 4 }, { 8, 16 } }, new short[] { 2, 4 }, device).Should().BeEquivalentTo(new short[,] { { 1, 1 }, { 4, 4 } });
        MatrixVectorTest<uint>(new uint[,] { { 2, 4 }, { 8, 16 } }, new uint[] { 2, 4 }, device).Should().BeEquivalentTo(new uint[,] { { 1, 1 }, { 4, 4 } });
        MatrixVectorTest<int>(new int[,] { { 2, 4 }, { 8, 16 } }, new int[] { 2, 4 }, device).Should().BeEquivalentTo(new int[,] { { 1, 1 }, { 4, 4 } });
        MatrixVectorTest<ulong>(new ulong[,] { { 2, 4 }, { 8, 16 } }, new ulong[] { 2, 4 }, device).Should().BeEquivalentTo(new ulong[,] { { 1, 1 }, { 4, 4 } });
        MatrixVectorTest<long>(new long[,] { { 2, 4 }, { 8, 16 } }, new long[] { 2, 4 }, device).Should().BeEquivalentTo(new long[,] { { 1, 1 }, { 4, 4 } });
    }

    private static Array MatrixVectorTest<TNumber>(TNumber[,] left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftMatrix = Tensor.FromArray<TNumber>(left).ToMatrix();
        using var rightVector = Tensor.FromArray<TNumber>(right).ToVector();

        leftMatrix.To(device);
        rightVector.To(device);

        using var result = leftMatrix.Divide(rightVector);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndMatrix(IDevice device)
    {
        MatrixMatrixTest<BFloat16>(new BFloat16[,] { { 2, 4 }, { 6, 8 } }, new BFloat16[,] { { 2, 4 }, { 6, 8 } }, device).Should().BeEquivalentTo(new BFloat16[,] { { 1, 1 }, { 1, 1 } });
        MatrixMatrixTest<float>(new float[,] { { 2, 4 }, { 6, 8 } }, new float[,] { { 2, 4 }, { 6, 8 } }, device).Should().BeEquivalentTo(new float[,] { { 1, 1 }, { 1, 1 } });
        MatrixMatrixTest<double>(new double[,] { { 2, 4 }, { 6, 8 } }, new double[,] { { 2, 4 }, { 6, 8 } }, device).Should().BeEquivalentTo(new double[,] { { 1, 1 }, { 1, 1 } });
        MatrixMatrixTest<byte>(new byte[,] { { 2, 4 }, { 6, 8 } }, new byte[,] { { 2, 4 }, { 6, 8 } }, device).Should().BeEquivalentTo(new byte[,] { { 1, 1 }, { 1, 1 } });
        MatrixMatrixTest<sbyte>(new sbyte[,] { { 2, 4 }, { 6, 8 } }, new sbyte[,] { { 2, 4 }, { 6, 8 } }, device).Should().BeEquivalentTo(new sbyte[,] { { 1, 1 }, { 1, 1 } });
        MatrixMatrixTest<ushort>(new ushort[,] { { 2, 4 }, { 6, 8 } }, new ushort[,] { { 2, 4 }, { 6, 8 } }, device).Should().BeEquivalentTo(new ushort[,] { { 1, 1 }, { 1, 1 } });
        MatrixMatrixTest<short>(new short[,] { { 2, 4 }, { 6, 8 } }, new short[,] { { 2, 4 }, { 6, 8 } }, device).Should().BeEquivalentTo(new short[,] { { 1, 1 }, { 1, 1 } });
        MatrixMatrixTest<uint>(new uint[,] { { 2, 4 }, { 6, 8 } }, new uint[,] { { 2, 4 }, { 6, 8 } }, device).Should().BeEquivalentTo(new uint[,] { { 1, 1 }, { 1, 1 } });
        MatrixMatrixTest<int>(new int[,] { { 2, 4 }, { 6, 8 } }, new int[,] { { 2, 4 }, { 6, 8 } }, device).Should().BeEquivalentTo(new int[,] { { 1, 1 }, { 1, 1 } });
        MatrixMatrixTest<ulong>(new ulong[,] { { 2, 4 }, { 6, 8 } }, new ulong[,] { { 2, 4 }, { 6, 8 } }, device).Should().BeEquivalentTo(new ulong[,] { { 1, 1 }, { 1, 1 } });
        MatrixMatrixTest<long>(new long[,] { { 2, 4 }, { 6, 8 } }, new long[,] { { 2, 4 }, { 6, 8 } }, device).Should().BeEquivalentTo(new long[,] { { 1, 1 }, { 1, 1 } });
    }

    private static Array MatrixMatrixTest<TNumber>(TNumber[,] left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftMatrix = Tensor.FromArray<TNumber>(left).ToMatrix();
        using var rightMatrix = Tensor.FromArray<TNumber>(right).ToMatrix();

        leftMatrix.To(device);
        rightMatrix.To(device);

        using var result = leftMatrix.Divide(rightMatrix);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndTensor(IDevice device)
    {
        MatrixTensorTest<BFloat16>(new BFloat16[,] { { 2, 4 }, { 6, 8 } }, new BFloat16[,,] { { { 2, 4 }, { 6, 8 } }, { { 2, 4 }, { 6, 8 } } }, device).Should().BeEquivalentTo(new BFloat16[,,] { { { 1, 1 }, { 1, 1 } }, { { 1, 1 }, { 1, 1 } } });
        MatrixTensorTest<float>(new float[,] { { 2, 4 }, { 6, 8 } }, new float[,,] { { { 2, 4 }, { 6, 8 } }, { { 2, 4 }, { 6, 8 } } }, device).Should().BeEquivalentTo(new float[,,] { { { 1, 1 }, { 1, 1 } }, { { 1, 1 }, { 1, 1 } } });
        MatrixTensorTest<double>(new double[,] { { 2, 4 }, { 6, 8 } }, new double[,,] { { { 2, 4 }, { 6, 8 } }, { { 2, 4 }, { 6, 8 } } }, device).Should().BeEquivalentTo(new double[,,] { { { 1, 1 }, { 1, 1 } }, { { 1, 1 }, { 1, 1 } } });
        MatrixTensorTest<byte>(new byte[,] { { 2, 4 }, { 6, 8 } }, new byte[,,] { { { 2, 4 }, { 6, 8 } }, { { 2, 4 }, { 6, 8 } } }, device).Should().BeEquivalentTo(new byte[,,] { { { 1, 1 }, { 1, 1 } }, { { 1, 1 }, { 1, 1 } } });
        MatrixTensorTest<sbyte>(new sbyte[,] { { 2, 4 }, { 6, 8 } }, new sbyte[,,] { { { 2, 4 }, { 6, 8 } }, { { 2, 4 }, { 6, 8 } } }, device).Should().BeEquivalentTo(new sbyte[,,] { { { 1, 1 }, { 1, 1 } }, { { 1, 1 }, { 1, 1 } } });
        MatrixTensorTest<ushort>(new ushort[,] { { 2, 4 }, { 6, 8 } }, new ushort[,,] { { { 2, 4 }, { 6, 8 } }, { { 2, 4 }, { 6, 8 } } }, device).Should().BeEquivalentTo(new ushort[,,] { { { 1, 1 }, { 1, 1 } }, { { 1, 1 }, { 1, 1 } } });
        MatrixTensorTest<short>(new short[,] { { 2, 4 }, { 6, 8 } }, new short[,,] { { { 2, 4 }, { 6, 8 } }, { { 2, 4 }, { 6, 8 } } }, device).Should().BeEquivalentTo(new short[,,] { { { 1, 1 }, { 1, 1 } }, { { 1, 1 }, { 1, 1 } } });
        MatrixTensorTest<uint>(new uint[,] { { 2, 4 }, { 6, 8 } }, new uint[,,] { { { 2, 4 }, { 6, 8 } }, { { 2, 4 }, { 6, 8 } } }, device).Should().BeEquivalentTo(new uint[,,] { { { 1, 1 }, { 1, 1 } }, { { 1, 1 }, { 1, 1 } } });
        MatrixTensorTest<int>(new int[,] { { 2, 4 }, { 6, 8 } }, new int[,,] { { { 2, 4 }, { 6, 8 } }, { { 2, 4 }, { 6, 8 } } }, device).Should().BeEquivalentTo(new int[,,] { { { 1, 1 }, { 1, 1 } }, { { 1, 1 }, { 1, 1 } } });
        MatrixTensorTest<ulong>(new ulong[,] { { 2, 4 }, { 6, 8 } }, new ulong[,,] { { { 2, 4 }, { 6, 8 } }, { { 2, 4 }, { 6, 8 } } }, device).Should().BeEquivalentTo(new ulong[,,] { { { 1, 1 }, { 1, 1 } }, { { 1, 1 }, { 1, 1 } } });
        MatrixTensorTest<long>(new long[,] { { 2, 4 }, { 6, 8 } }, new long[,,] { { { 2, 4 }, { 6, 8 } }, { { 2, 4 }, { 6, 8 } } }, device).Should().BeEquivalentTo(new long[,,] { { { 1, 1 }, { 1, 1 } }, { { 1, 1 }, { 1, 1 } } });
    }

    private static Array MatrixTensorTest<TNumber>(TNumber[,] left, TNumber[,,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftMatrix = Tensor.FromArray<TNumber>(left).ToMatrix();
        using var rightTensor = Tensor.FromArray<TNumber>(right).ToTensor();

        leftMatrix.To(device);
        rightTensor.To(device);

        using var result = leftMatrix.Divide(rightTensor);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorAndScalar(IDevice device)
    {
        TensorScalarTest<BFloat16>(new BFloat16[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, 2, device).Should().BeEquivalentTo(new BFloat16[,,] { { { 1, 2 }, { 4, 8 } }, { { 1, 2 }, { 4, 8 } } });
        TensorScalarTest<float>(new float[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, 2, device).Should().BeEquivalentTo(new float[,,] { { { 1, 2 }, { 4, 8 } }, { { 1, 2 }, { 4, 8 } } });
        TensorScalarTest<double>(new double[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, 2, device).Should().BeEquivalentTo(new double[,,] { { { 1, 2 }, { 4, 8 } }, { { 1, 2 }, { 4, 8 } } });
        TensorScalarTest<byte>(new byte[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, 2, device).Should().BeEquivalentTo(new byte[,,] { { { 1, 2 }, { 4, 8 } }, { { 1, 2 }, { 4, 8 } } });
        TensorScalarTest<sbyte>(new sbyte[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, 2, device).Should().BeEquivalentTo(new sbyte[,,] { { { 1, 2 }, { 4, 8 } }, { { 1, 2 }, { 4, 8 } } });
        TensorScalarTest<ushort>(new ushort[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, 2, device).Should().BeEquivalentTo(new ushort[,,] { { { 1, 2 }, { 4, 8 } }, { { 1, 2 }, { 4, 8 } } });
        TensorScalarTest<short>(new short[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, 2, device).Should().BeEquivalentTo(new short[,,] { { { 1, 2 }, { 4, 8 } }, { { 1, 2 }, { 4, 8 } } });
        TensorScalarTest<uint>(new uint[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, 2, device).Should().BeEquivalentTo(new uint[,,] { { { 1, 2 }, { 4, 8 } }, { { 1, 2 }, { 4, 8 } } });
        TensorScalarTest<int>(new int[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, 2, device).Should().BeEquivalentTo(new int[,,] { { { 1, 2 }, { 4, 8 } }, { { 1, 2 }, { 4, 8 } } });
        TensorScalarTest<ulong>(new ulong[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, 2, device).Should().BeEquivalentTo(new ulong[,,] { { { 1, 2 }, { 4, 8 } }, { { 1, 2 }, { 4, 8 } } });
        TensorScalarTest<long>(new long[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, 2, device).Should().BeEquivalentTo(new long[,,] { { { 1, 2 }, { 4, 8 } }, { { 1, 2 }, { 4, 8 } } });
    }

    private static Array TensorScalarTest<TNumber>(TNumber[,,] left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftTensor = Tensor.FromArray<TNumber>(left).ToTensor();
        using var rightScalar = new Scalar<TNumber>(right);

        leftTensor.To(device);
        rightScalar.To(device);

        using var result = leftTensor.Divide(rightScalar);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorAndVector(IDevice device)
    {
        TensorVectorTest<BFloat16>(new BFloat16[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, new BFloat16[] { 2, 4 }, device).Should().BeEquivalentTo(new BFloat16[,,] { { { 1, 1 }, { 4, 4 } }, { { 1, 1 }, { 4, 4 } } });
        TensorVectorTest<float>(new float[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, new float[] { 2, 4 }, device).Should().BeEquivalentTo(new float[,,] { { { 1, 1 }, { 4, 4 } }, { { 1, 1 }, { 4, 4 } } });
        TensorVectorTest<double>(new double[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, new double[] { 2, 4 }, device).Should().BeEquivalentTo(new double[,,] { { { 1, 1 }, { 4, 4 } }, { { 1, 1 }, { 4, 4 } } });
        TensorVectorTest<byte>(new byte[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, new byte[] { 2, 4 }, device).Should().BeEquivalentTo(new byte[,,] { { { 1, 1 }, { 4, 4 } }, { { 1, 1 }, { 4, 4 } } });
        TensorVectorTest<sbyte>(new sbyte[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, new sbyte[] { 2, 4 }, device).Should().BeEquivalentTo(new sbyte[,,] { { { 1, 1 }, { 4, 4 } }, { { 1, 1 }, { 4, 4 } } });
        TensorVectorTest<ushort>(new ushort[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, new ushort[] { 2, 4 }, device).Should().BeEquivalentTo(new ushort[,,] { { { 1, 1 }, { 4, 4 } }, { { 1, 1 }, { 4, 4 } } });
        TensorVectorTest<short>(new short[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, new short[] { 2, 4 }, device).Should().BeEquivalentTo(new short[,,] { { { 1, 1 }, { 4, 4 } }, { { 1, 1 }, { 4, 4 } } });
        TensorVectorTest<uint>(new uint[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, new uint[] { 2, 4 }, device).Should().BeEquivalentTo(new uint[,,] { { { 1, 1 }, { 4, 4 } }, { { 1, 1 }, { 4, 4 } } });
        TensorVectorTest<int>(new int[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, new int[] { 2, 4 }, device).Should().BeEquivalentTo(new int[,,] { { { 1, 1 }, { 4, 4 } }, { { 1, 1 }, { 4, 4 } } });
        TensorVectorTest<ulong>(new ulong[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, new ulong[] { 2, 4 }, device).Should().BeEquivalentTo(new ulong[,,] { { { 1, 1 }, { 4, 4 } }, { { 1, 1 }, { 4, 4 } } });
        TensorVectorTest<long>(new long[,,] { { { 2, 4 }, { 8, 16 } }, { { 2, 4 }, { 8, 16 } } }, new long[] { 2, 4 }, device).Should().BeEquivalentTo(new long[,,] { { { 1, 1 }, { 4, 4 } }, { { 1, 1 }, { 4, 4 } } });
    }

    private static Array TensorVectorTest<TNumber>(TNumber[,,] left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftTensor = Tensor.FromArray<TNumber>(left).ToTensor();
        using var rightVector = Tensor.FromArray<TNumber>(right).ToVector();

        leftTensor.To(device);
        rightVector.To(device);

        using var result = leftTensor.Divide(rightVector);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorAndMatrix(IDevice device)
    {
        TensorMatrixTest<BFloat16>(new BFloat16[,,] { { { 2, 4 }, { 8, 16 } } }, new BFloat16[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new BFloat16[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorMatrixTest<float>(new float[,,] { { { 2, 4 }, { 8, 16 } } }, new float[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new float[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorMatrixTest<double>(new double[,,] { { { 2, 4 }, { 8, 16 } } }, new double[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new double[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorMatrixTest<byte>(new byte[,,] { { { 2, 4 }, { 8, 16 } } }, new byte[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new byte[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorMatrixTest<sbyte>(new sbyte[,,] { { { 2, 4 }, { 8, 16 } } }, new sbyte[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new sbyte[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorMatrixTest<ushort>(new ushort[,,] { { { 2, 4 }, { 8, 16 } } }, new ushort[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new ushort[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorMatrixTest<short>(new short[,,] { { { 2, 4 }, { 8, 16 } } }, new short[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new short[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorMatrixTest<uint>(new uint[,,] { { { 2, 4 }, { 8, 16 } } }, new uint[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new uint[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorMatrixTest<int>(new int[,,] { { { 2, 4 }, { 8, 16 } } }, new int[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new int[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorMatrixTest<ulong>(new ulong[,,] { { { 2, 4 }, { 8, 16 } } }, new ulong[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new ulong[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorMatrixTest<long>(new long[,,] { { { 2, 4 }, { 8, 16 } } }, new long[,] { { 2, 4 }, { 8, 16 } }, device).Should().BeEquivalentTo(new long[,,] { { { 1, 1 }, { 1, 1 } } });
    }

    private static Array TensorMatrixTest<TNumber>(TNumber[,,] left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftTensor = Tensor.FromArray<TNumber>(left).ToTensor();
        using var rightMatrix = Tensor.FromArray<TNumber>(right).ToMatrix();

        leftTensor.To(device);
        rightMatrix.To(device);

        using var result = leftTensor.Divide(rightMatrix);

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensorAndTensor(IDevice device)
    {
        TensorTensorTest<BFloat16>(new BFloat16[,,] { { { 2, 4 }, { 8, 16 } } }, new BFloat16[,,] { { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new BFloat16[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorTensorTest<float>(new float[,,] { { { 2, 4 }, { 8, 16 } } }, new float[,,] { { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new float[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorTensorTest<double>(new double[,,] { { { 2, 4 }, { 8, 16 } } }, new double[,,] { { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new double[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorTensorTest<byte>(new byte[,,] { { { 2, 4 }, { 8, 16 } } }, new byte[,,] { { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new byte[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorTensorTest<sbyte>(new sbyte[,,] { { { 2, 4 }, { 8, 16 } } }, new sbyte[,,] { { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new sbyte[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorTensorTest<ushort>(new ushort[,,] { { { 2, 4 }, { 8, 16 } } }, new ushort[,,] { { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new ushort[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorTensorTest<short>(new short[,,] { { { 2, 4 }, { 8, 16 } } }, new short[,,] { { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new short[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorTensorTest<uint>(new uint[,,] { { { 2, 4 }, { 8, 16 } } }, new uint[,,] { { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new uint[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorTensorTest<int>(new int[,,] { { { 2, 4 }, { 8, 16 } } }, new int[,,] { { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new int[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorTensorTest<ulong>(new ulong[,,] { { { 2, 4 }, { 8, 16 } } }, new ulong[,,] { { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new ulong[,,] { { { 1, 1 }, { 1, 1 } } });
        TensorTensorTest<long>(new long[,,] { { { 2, 4 }, { 8, 16 } } }, new long[,,] { { { 2, 4 }, { 8, 16 } } }, device).Should().BeEquivalentTo(new long[,,] { { { 1, 1 }, { 1, 1 } } });
    }

    private static Array TensorTensorTest<TNumber>(TNumber[,,] left, TNumber[,,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var leftTensor = Tensor.FromArray<TNumber>(left).ToTensor();
        using var rightTensor = Tensor.FromArray<TNumber>(right).ToTensor();

        leftTensor.To(device);
        rightTensor.To(device);

        using var result = leftTensor.Divide(rightTensor);

        return result.ToArray();
    }
}