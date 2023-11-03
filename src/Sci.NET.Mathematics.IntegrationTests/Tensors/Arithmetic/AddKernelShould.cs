// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Arithmetic;

public class AddKernelShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarsAndScalar(IDevice device)
    {
        ScalarScalarTest<float>(1, 2, device).Should().Be(3);
        ScalarScalarTest<double>(1, 2, device).Should().Be(3);
        ScalarScalarTest<byte>(1, 2, device).Should().Be(3);
        ScalarScalarTest<sbyte>(1, 2, device).Should().Be(3);
        ScalarScalarTest<ushort>(1, 2, device).Should().Be(3);
        ScalarScalarTest<short>(1, 2, device).Should().Be(3);
        ScalarScalarTest<uint>(1, 2, device).Should().Be(3);
        ScalarScalarTest<int>(1, 2, device).Should().Be(3);
        ScalarScalarTest<ulong>(1, 2, device).Should().Be(3);
        ScalarScalarTest<long>(1, 2, device).Should().Be(3);
        ScalarScalarTest<BFloat16>(1, 2, device).Should().Be(3);
    }

    private static TNumber ScalarScalarTest<TNumber>(TNumber left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = new Scalar<TNumber>(left);
        var rightScalar = new Scalar<TNumber>(right);
        leftScalar.To(device);
        rightScalar.To(device);

        var resultScalar = leftScalar.Add(rightScalar);

        return resultScalar.Value;
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarAndVector(IDevice device)
    {
        ScalarVectorTest<float>(1, new float[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new float[] { 2, 3, 4, 5 });
        ScalarVectorTest<double>(1, new double[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new double[] { 2, 3, 4, 5 });
        ScalarVectorTest<byte>(1, new byte[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new byte[] { 2, 3, 4, 5 });
        ScalarVectorTest<sbyte>(1, new sbyte[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new sbyte[] { 2, 3, 4, 5 });
        ScalarVectorTest<ushort>(1, new ushort[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new ushort[] { 2, 3, 4, 5 });
        ScalarVectorTest<short>(1, new short[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new short[] { 2, 3, 4, 5 });
        ScalarVectorTest<uint>(1, new uint[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new uint[] { 2, 3, 4, 5 });
        ScalarVectorTest<int>(1, new int[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new int[] { 2, 3, 4, 5 });
        ScalarVectorTest<ulong>(1, new ulong[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new ulong[] { 2, 3, 4, 5 });
        ScalarVectorTest<long>(1, new long[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new long[] { 2, 3, 4, 5 });
        ScalarVectorTest<BFloat16>(1, new BFloat16[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new BFloat16[] { 2, 3, 4, 5 });
    }

    private static Array ScalarVectorTest<TNumber>(TNumber left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = new Scalar<TNumber>(left);
        var rightVector = Tensor.FromArray<TNumber>(right).ToVector();
        leftScalar.To(device);
        rightVector.To(device);

        var result = leftScalar.Add(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarAndMatrix(IDevice device)
    {
        ScalarMatrixTest<float>(1, new float[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new float[,] { { 2, 3 }, { 4, 5 } });
        ScalarMatrixTest<double>(1, new double[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new double[,] { { 2, 3 }, { 4, 5 } });
        ScalarMatrixTest<byte>(1, new byte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new byte[,] { { 2, 3 }, { 4, 5 } });
        ScalarMatrixTest<sbyte>(1, new sbyte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new sbyte[,] { { 2, 3 }, { 4, 5 } });
        ScalarMatrixTest<ushort>(1, new ushort[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ushort[,] { { 2, 3 }, { 4, 5 } });
        ScalarMatrixTest<short>(1, new short[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new short[,] { { 2, 3 }, { 4, 5 } });
        ScalarMatrixTest<uint>(1, new uint[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new uint[,] { { 2, 3 }, { 4, 5 } });
        ScalarMatrixTest<int>(1, new int[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new int[,] { { 2, 3 }, { 4, 5 } });
        ScalarMatrixTest<ulong>(1, new ulong[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ulong[,] { { 2, 3 }, { 4, 5 } });
        ScalarMatrixTest<long>(1, new long[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new long[,] { { 2, 3 }, { 4, 5 } });
        ScalarMatrixTest<BFloat16>(1, new BFloat16[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new BFloat16[,] { { 2, 3 }, { 4, 5 } });
    }

    private static Array ScalarMatrixTest<TNumber>(TNumber left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = new Scalar<TNumber>(left);
        var rightVector = Tensor.FromArray<TNumber>(right).ToMatrix();
        leftScalar.To(device);
        rightVector.To(device);

        var result = leftScalar.Add(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalarTensor(IDevice device)
    {
        ScalarTensorTest<float>(1, new float[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device)
            .Should()
            .BeEquivalentTo(new float[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });

        ScalarTensorTest<double>(1, new double[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device)
            .Should()
            .BeEquivalentTo(new double[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });

        ScalarTensorTest<byte>(1, new byte[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device)
            .Should()
            .BeEquivalentTo(new byte[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });

        ScalarTensorTest<sbyte>(1, new sbyte[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device)
            .Should()
            .BeEquivalentTo(new sbyte[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });

        ScalarTensorTest<sbyte>(1, new sbyte[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device)
            .Should()
            .BeEquivalentTo(new sbyte[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });

        ScalarTensorTest<byte>(1, new byte[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device)
            .Should()
            .BeEquivalentTo(new byte[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });

        ScalarTensorTest<sbyte>(1, new sbyte[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device)
            .Should()
            .BeEquivalentTo(new sbyte[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });

        ScalarTensorTest<ushort>(1, new ushort[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device)
            .Should()
            .BeEquivalentTo(new ushort[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });

        ScalarTensorTest<short>(1, new short[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device)
            .Should()
            .BeEquivalentTo(new short[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });

        ScalarTensorTest<uint>(1, new uint[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device)
            .Should()
            .BeEquivalentTo(new uint[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });

        ScalarTensorTest<int>(1, new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device)
            .Should()
            .BeEquivalentTo(new int[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });

        ScalarTensorTest<ulong>(1, new ulong[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device)
            .Should()
            .BeEquivalentTo(new ulong[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });

        ScalarTensorTest<long>(1, new long[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device)
            .Should()
            .BeEquivalentTo(new long[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });

        ScalarTensorTest<BFloat16>(1, new BFloat16[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, device)
            .Should()
            .BeEquivalentTo(new BFloat16[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });
    }

    private static Array ScalarTensorTest<TNumber>(TNumber left, TNumber[,,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = new Scalar<TNumber>(left);
        var rightVector = Tensor.FromArray<TNumber>(right).ToTensor();
        leftScalar.To(device);
        rightVector.To(device);

        var result = leftScalar.Add(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndScalar(IDevice device)
    {
        VectorScalarTest<float>(new float[] { 1, 2, 3, 4 }, 1, device).Should().BeEquivalentTo(new float[] { 2, 3, 4, 5 });
        VectorScalarTest<double>(new double[] { 1, 2, 3, 4 }, 1, device).Should().BeEquivalentTo(new double[] { 2, 3, 4, 5 });
        VectorScalarTest<byte>(new byte[] { 1, 2, 3, 4 }, 1, device).Should().BeEquivalentTo(new byte[] { 2, 3, 4, 5 });
        VectorScalarTest<sbyte>(new sbyte[] { 1, 2, 3, 4 }, 1, device).Should().BeEquivalentTo(new sbyte[] { 2, 3, 4, 5 });
        VectorScalarTest<ushort>(new ushort[] { 1, 2, 3, 4 }, 1, device).Should().BeEquivalentTo(new ushort[] { 2, 3, 4, 5 });
        VectorScalarTest<short>(new short[] { 1, 2, 3, 4 }, 1, device).Should().BeEquivalentTo(new short[] { 2, 3, 4, 5 });
        VectorScalarTest<uint>(new uint[] { 1, 2, 3, 4 }, 1, device).Should().BeEquivalentTo(new uint[] { 2, 3, 4, 5 });
        VectorScalarTest<int>(new int[] { 1, 2, 3, 4 }, 1, device).Should().BeEquivalentTo(new int[] { 2, 3, 4, 5 });
        VectorScalarTest<ulong>(new ulong[] { 1, 2, 3, 4 }, 1, device).Should().BeEquivalentTo(new ulong[] { 2, 3, 4, 5 });
        VectorScalarTest<long>(new long[] { 1, 2, 3, 4 }, 1, device).Should().BeEquivalentTo(new long[] { 2, 3, 4, 5 });
        VectorScalarTest<BFloat16>(new BFloat16[] { 1, 2, 3, 4 }, 1, device).Should().BeEquivalentTo(new BFloat16[] { 2, 3, 4, 5 });
    }

    private static Array VectorScalarTest<TNumber>(TNumber[] left, TNumber right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = Tensor.FromArray<TNumber>(left).ToVector();
        var rightVector = new Scalar<TNumber>(right);
        leftScalar.To(device);
        rightVector.To(device);

        var result = leftScalar.Add(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndVector(IDevice device)
    {
        VectorVectorTest<float>(new float[] { 1, 2, 3, 4 }, new float[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new float[] { 2, 4, 6, 8 });
        VectorVectorTest<double>(new double[] { 1, 2, 3, 4 }, new double[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new double[] { 2, 4, 6, 8 });
        VectorVectorTest<byte>(new byte[] { 1, 2, 3, 4 }, new byte[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new byte[] { 2, 4, 6, 8 });
        VectorVectorTest<sbyte>(new sbyte[] { 1, 2, 3, 4 }, new sbyte[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new sbyte[] { 2, 4, 6, 8 });
        VectorVectorTest<ushort>(new ushort[] { 1, 2, 3, 4 }, new ushort[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new ushort[] { 2, 4, 6, 8 });
        VectorVectorTest<short>(new short[] { 1, 2, 3, 4 }, new short[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new short[] { 2, 4, 6, 8 });
        VectorVectorTest<uint>(new uint[] { 1, 2, 3, 4 }, new uint[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new uint[] { 2, 4, 6, 8 });
        VectorVectorTest<int>(new int[] { 1, 2, 3, 4 }, new int[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new int[] { 2, 4, 6, 8 });
        VectorVectorTest<ulong>(new ulong[] { 1, 2, 3, 4 }, new ulong[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new ulong[] { 2, 4, 6, 8 });
        VectorVectorTest<long>(new long[] { 1, 2, 3, 4 }, new long[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new long[] { 2, 4, 6, 8 });
        VectorVectorTest<BFloat16>(new BFloat16[] { 1, 2, 3, 4 }, new BFloat16[] { 1, 2, 3, 4 }, device).Should().BeEquivalentTo(new BFloat16[] { 2, 4, 6, 8 });
    }

    private static Array VectorVectorTest<TNumber>(TNumber[] left, TNumber[] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = Tensor.FromArray<TNumber>(left).ToVector();
        var rightVector = Tensor.FromArray<TNumber>(right).ToVector();
        leftScalar.To(device);
        rightVector.To(device);

        var result = leftScalar.Add(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVectorAndMatrix(IDevice device)
    {
        VectorMatrixTest<float>(new float[] { 1, 2, 3, 4 }, new float[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new float[,] { { 2, 4 }, { 6, 8 } });
        VectorMatrixTest<double>(new double[] { 1, 2, 3, 4 }, new double[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new double[,] { { 2, 4 }, { 6, 8 } });
        VectorMatrixTest<byte>(new byte[] { 1, 2, 3, 4 }, new byte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new byte[,] { { 2, 4 }, { 6, 8 } });
        VectorMatrixTest<sbyte>(new sbyte[] { 1, 2, 3, 4 }, new sbyte[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new sbyte[,] { { 2, 4 }, { 6, 8 } });
        VectorMatrixTest<ushort>(new ushort[] { 1, 2, 3, 4 }, new ushort[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ushort[,] { { 2, 4 }, { 6, 8 } });
        VectorMatrixTest<short>(new short[] { 1, 2, 3, 4 }, new short[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new short[,] { { 2, 4 }, { 6, 8 } });
        VectorMatrixTest<uint>(new uint[] { 1, 2, 3, 4 }, new uint[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new uint[,] { { 2, 4 }, { 6, 8 } });
        VectorMatrixTest<int>(new int[] { 1, 2, 3, 4 }, new int[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new int[,] { { 2, 4 }, { 6, 8 } });
        VectorMatrixTest<ulong>(new ulong[] { 1, 2, 3, 4 }, new ulong[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new ulong[,] { { 2, 4 }, { 6, 8 } });
        VectorMatrixTest<long>(new long[] { 1, 2, 3, 4 }, new long[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new long[,] { { 2, 4 }, { 6, 8 } });
        VectorMatrixTest<BFloat16>(new BFloat16[] { 1, 2, 3, 4 }, new BFloat16[,] { { 1, 2 }, { 3, 4 } }, device).Should().BeEquivalentTo(new BFloat16[,] { { 2, 4 }, { 6, 8 } });
    }

    private static Array VectorMatrixTest<TNumber>(TNumber[] left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = Tensor.FromArray<TNumber>(left).ToVector();
        var rightVector = Tensor.FromArray<TNumber>(right).ToVector();
        leftScalar.To(device);
        rightVector.To(device);

        var result = leftScalar.Add(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }
}