// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Arithmetic;

public class AbsShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalar(IDevice device)
    {
        AbsScalarTest<float>(-1, device).Should().Be(1);
        AbsScalarTest<double>(-1, device).Should().Be(1);
        AbsScalarTest<sbyte>(-1, device).Should().Be(1);
        AbsScalarTest<byte>(1, device).Should().Be(1);
        AbsScalarTest<short>(-1, device).Should().Be(1);
        AbsScalarTest<ushort>(1, device).Should().Be(1);
        AbsScalarTest<int>(-1, device).Should().Be(1);
        AbsScalarTest<uint>(1, device).Should().Be(1);
        AbsScalarTest<long>(-1, device).Should().Be(1);
        AbsScalarTest<ulong>(1, device).Should().Be(1);
    }

    private static TNumber AbsScalarTest<TNumber>(TNumber number, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var scalar = new Scalar<TNumber>(number);

        scalar.To(device);

        var result = scalar.Abs();

        return result.Value;
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVector(IDevice device)
    {
        AbsVectorTest<float>(new float[] { -1, -2, -3 }, device).Should().BeEquivalentTo(new float[] { 1, 2, 3 });
        AbsVectorTest<double>(new double[] { -1, -2, -3 }, device).Should().BeEquivalentTo(new double[] { 1, 2, 3 });
        AbsVectorTest<sbyte>(new sbyte[] { -1, -2, -3 }, device).Should().BeEquivalentTo(new sbyte[] { 1, 2, 3 });
        AbsVectorTest<byte>(new byte[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new byte[] { 1, 2, 3 });
        AbsVectorTest<short>(new short[] { -1, -2, -3 }, device).Should().BeEquivalentTo(new short[] { 1, 2, 3 });
        AbsVectorTest<ushort>(new ushort[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new ushort[] { 1, 2, 3 });
        AbsVectorTest<int>(new int[] { -1, -2, -3 }, device).Should().BeEquivalentTo(new int[] { 1, 2, 3 });
        AbsVectorTest<uint>(new uint[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new uint[] { 1, 2, 3 });
        AbsVectorTest<long>(new long[] { -1, -2, -3 }, device).Should().BeEquivalentTo(new long[] { 1, 2, 3 });
        AbsVectorTest<ulong>(new ulong[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new ulong[] { 1, 2, 3 });
    }

    private static Array AbsVectorTest<TNumber>(TNumber[] numbers, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var vector = Tensor.FromArray<TNumber>(numbers).ToVector();

        vector.To(device);

        var result = vector.Abs();

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrix(IDevice device)
    {
        AbsMatrixTest<float>(new float[,] { { -1, -2, -3 }, { -4, -5, -6 } }, device).Should().BeEquivalentTo(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        AbsMatrixTest<double>(new double[,] { { -1, -2, -3 }, { -4, -5, -6 } }, device).Should().BeEquivalentTo(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        AbsMatrixTest<sbyte>(new sbyte[,] { { -1, -2, -3 }, { -4, -5, -6 } }, device).Should().BeEquivalentTo(new sbyte[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        AbsMatrixTest<byte>(new byte[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new byte[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        AbsMatrixTest<short>(new short[,] { { -1, -2, -3 }, { -4, -5, -6 } }, device).Should().BeEquivalentTo(new short[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        AbsMatrixTest<ushort>(new ushort[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new ushort[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        AbsMatrixTest<int>(new int[,] { { -1, -2, -3 }, { -4, -5, -6 } }, device).Should().BeEquivalentTo(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        AbsMatrixTest<uint>(new uint[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new uint[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        AbsMatrixTest<long>(new long[,] { { -1, -2, -3 }, { -4, -5, -6 } }, device).Should().BeEquivalentTo(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        AbsMatrixTest<ulong>(new ulong[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new ulong[,] { { 1, 2, 3 }, { 4, 5, 6 } });
    }

    private static Array AbsMatrixTest<TNumber>(TNumber[,] numbers, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var matrix = Tensor.FromArray<TNumber>(numbers).ToMatrix();

        matrix.To(device);

        var result = matrix.Abs();

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensor(IDevice device)
    {
        AbsTensorTest<float>(new float[,,] { { { -1, -2, -3 }, { -4, -5, -6 } }, { { -7, -8, -9 }, { -10, -11, -12 } } }, device).Should().BeEquivalentTo(new float[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } });
        AbsTensorTest<double>(new double[,,] { { { -1, -2, -3 }, { -4, -5, -6 } }, { { -7, -8, -9 }, { -10, -11, -12 } } }, device).Should().BeEquivalentTo(new double[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } });
        AbsTensorTest<sbyte>(new sbyte[,,] { { { -1, -2, -3 }, { -4, -5, -6 } }, { { -7, -8, -9 }, { -10, -11, -12 } } }, device).Should().BeEquivalentTo(new sbyte[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } });
        AbsTensorTest<byte>(new byte[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, device).Should().BeEquivalentTo(new byte[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } });
        AbsTensorTest<short>(new short[,,] { { { -1, -2, -3, -4 }, { -5, -6, -7, -8 } }, { { -9, -10, -11, -12 }, { -13, -14, -15, -16 } } }, device).Should().BeEquivalentTo(new short[,,] { { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }, { { 9, 10, 11, 12 }, { 13, 14, 15, 16 } } });
        AbsTensorTest<ushort>(new ushort[,,] { { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }, { { 9, 10, 11, 12 }, { 13, 14, 15, 16 } } }, device).Should().BeEquivalentTo(new ushort[,,] { { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }, { { 9, 10, 11, 12 }, { 13, 14, 15, 16 } } });
        AbsTensorTest<int>(new int[,,] { { { -1, -2, -3, -4 }, { -5, -6, -7, -8 } }, { { -9, -10, -11, -12 }, { -13, -14, -15, -16 } } }, device).Should().BeEquivalentTo(new int[,,] { { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }, { { 9, 10, 11, 12 }, { 13, 14, 15, 16 } } });
        AbsTensorTest<uint>(new uint[,,] { { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }, { { 9, 10, 11, 12 }, { 13, 14, 15, 16 } } }, device).Should().BeEquivalentTo(new uint[,,] { { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }, { { 9, 10, 11, 12 }, { 13, 14, 15, 16 } } });
        AbsTensorTest<long>(new long[,,] { { { -1, -2, -3, -4 }, { -5, -6, -7, -8 } }, { { -9, -10, -11, -12 }, { -13, -14, -15, -16 } } }, device).Should().BeEquivalentTo(new long[,,] { { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }, { { 9, 10, 11, 12 }, { 13, 14, 15, 16 } } });
        AbsTensorTest<ulong>(new ulong[,,] { { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }, { { 9, 10, 11, 12 }, { 13, 14, 15, 16 } } }, device).Should().BeEquivalentTo(new ulong[,,] { { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }, { { 9, 10, 11, 12 }, { 13, 14, 15, 16 } } });
    }

    private static Array AbsTensorTest<TNumber>(TNumber[,,] numbers, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensor = Tensor.FromArray<TNumber>(numbers).ToTensor();

        tensor.To(device);

        var result = tensor.Abs();

        return result.ToArray();
    }
}