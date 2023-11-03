// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.CUDA.Tensors;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.CUDA.UnitTests.Tensors.Arithmetic.Add;

public class CudaArithmeticAddKernelShould
{
    [Fact]
    public void ReturnExpectedResult_GivenScalarsAndScalar()
    {
        ScalarScalarTest<float>(1, 2).Should().Be(3);
        ScalarScalarTest<double>(1, 2).Should().Be(3);
        ScalarScalarTest<byte>(1, 2).Should().Be(3);
        ScalarScalarTest<sbyte>(1, 2).Should().Be(3);
        ScalarScalarTest<ushort>(1, 2).Should().Be(3);
        ScalarScalarTest<short>(1, 2).Should().Be(3);
        ScalarScalarTest<uint>(1, 2).Should().Be(3);
        ScalarScalarTest<int>(1, 2).Should().Be(3);
        ScalarScalarTest<ulong>(1, 2).Should().Be(3);
        ScalarScalarTest<long>(1, 2).Should().Be(3);
    }

    private static TNumber ScalarScalarTest<TNumber>(TNumber left, TNumber right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = new Scalar<TNumber>(left);
        var rightScalar = new Scalar<TNumber>(right);
        leftScalar.To<CudaComputeDevice>();
        rightScalar.To<CudaComputeDevice>();

        var resultScalar = leftScalar.Add(rightScalar);
        resultScalar.To<CpuComputeDevice>();

        return resultScalar.Value;
    }

    [Fact]
    public void ReturnExpectedResult_GivenScalarAndVector()
    {
        ScalarVectorTest<float>(1, new float[] { 1, 2, 3, 4 }).Should().BeEquivalentTo(new float[] { 2, 3, 4, 5 });
        ScalarVectorTest<double>(1, new double[] { 1, 2, 3, 4 }).Should().BeEquivalentTo(new double[] { 2, 3, 4, 5 });
        ScalarVectorTest<byte>(1, new byte[] { 1, 2, 3, 4 }).Should().BeEquivalentTo(new byte[] { 2, 3, 4, 5 });
        ScalarVectorTest<sbyte>(1, new sbyte[] { 1, 2, 3, 4 }).Should().BeEquivalentTo(new sbyte[] { 2, 3, 4, 5 });
        ScalarVectorTest<ushort>(1, new ushort[] { 1, 2, 3, 4 }).Should().BeEquivalentTo(new ushort[] { 2, 3, 4, 5 });
        ScalarVectorTest<short>(1, new short[] { 1, 2, 3, 4 }).Should().BeEquivalentTo(new short[] { 2, 3, 4, 5 });
        ScalarVectorTest<uint>(1, new uint[] { 1, 2, 3, 4 }).Should().BeEquivalentTo(new uint[] { 2, 3, 4, 5 });
        ScalarVectorTest<int>(1, new int[] { 1, 2, 3, 4 }).Should().BeEquivalentTo(new int[] { 2, 3, 4, 5 });
        ScalarVectorTest<ulong>(1, new ulong[] { 1, 2, 3, 4 }).Should().BeEquivalentTo(new ulong[] { 2, 3, 4, 5 });
        ScalarVectorTest<long>(1, new long[] { 1, 2, 3, 4 }).Should().BeEquivalentTo(new long[] { 2, 3, 4, 5 });
    }

    private static Array ScalarVectorTest<TNumber>(TNumber left, TNumber[] right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = new Scalar<TNumber>(left);
        var rightVector = Tensor.FromArray<TNumber>(right).ToVector();
        leftScalar.To<CudaComputeDevice>();
        rightVector.To<CudaComputeDevice>();

        var result = leftScalar.Add(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Fact]
    public void ReturnExpectedResult_GivenScalarAndMatrix()
    {
        ScalaMatrixTest<float>(1, new float[,] { { 1, 2 }, { 3, 4 } }).Should().BeEquivalentTo(new float[,] { { 2, 3 }, { 4, 5 } });
        ScalaMatrixTest<double>(1, new double[,] { { 1, 2 }, { 3, 4 } }).Should().BeEquivalentTo(new double[,] { { 2, 3 }, { 4, 5 } });
        ScalaMatrixTest<byte>(1, new byte[,] { { 1, 2 }, { 3, 4 } }).Should().BeEquivalentTo(new byte[,] { { 2, 3 }, { 4, 5 } });
        ScalaMatrixTest<sbyte>(1, new sbyte[,] { { 1, 2 }, { 3, 4 } }).Should().BeEquivalentTo(new sbyte[,] { { 2, 3 }, { 4, 5 } });
        ScalaMatrixTest<ushort>(1, new ushort[,] { { 1, 2 }, { 3, 4 } }).Should().BeEquivalentTo(new ushort[,] { { 2, 3 }, { 4, 5 } });
        ScalaMatrixTest<short>(1, new short[,] { { 1, 2 }, { 3, 4 } }).Should().BeEquivalentTo(new short[,] { { 2, 3 }, { 4, 5 } });
        ScalaMatrixTest<uint>(1, new uint[,] { { 1, 2 }, { 3, 4 } }).Should().BeEquivalentTo(new uint[,] { { 2, 3 }, { 4, 5 } });
        ScalaMatrixTest<int>(1, new int[,] { { 1, 2 }, { 3, 4 } }).Should().BeEquivalentTo(new int[,] { { 2, 3 }, { 4, 5 } });
        ScalaMatrixTest<ulong>(1, new ulong[,] { { 1, 2 }, { 3, 4 } }).Should().BeEquivalentTo(new ulong[,] { { 2, 3 }, { 4, 5 } });
        ScalaMatrixTest<long>(1, new long[,] { { 1, 2 }, { 3, 4 } }).Should().BeEquivalentTo(new long[,] { { 2, 3 }, { 4, 5 } });
    }

    private static Array ScalaMatrixTest<TNumber>(TNumber left, TNumber[,] right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = new Scalar<TNumber>(left);
        var rightVector = Tensor.FromArray<TNumber>(right).ToMatrix();
        leftScalar.To<CudaComputeDevice>();
        rightVector.To<CudaComputeDevice>();

        var result = leftScalar.Add(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }
}