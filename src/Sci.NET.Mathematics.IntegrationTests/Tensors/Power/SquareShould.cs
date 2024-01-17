// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Power;

public class SquareShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenScalar(IDevice device)
    {
        SquareScalarTest<float>(2, device).Should().Be(4);
        SquareScalarTest<double>(2, device).Should().Be(4);
        SquareScalarTest<sbyte>(2, device).Should().Be(4);
        SquareScalarTest<byte>(2, device).Should().Be(4);
        SquareScalarTest<short>(2, device).Should().Be(4);
        SquareScalarTest<ushort>(2, device).Should().Be(4);
        SquareScalarTest<int>(2, device).Should().Be(4);
        SquareScalarTest<uint>(2, device).Should().Be(4);
        SquareScalarTest<long>(2, device).Should().Be(4);
        SquareScalarTest<ulong>(2, device).Should().Be(4);
    }

    private static TNumber SquareScalarTest<TNumber>(TNumber number, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var scalar = new Scalar<TNumber>(number);

        scalar.To(device);

        var result = scalar.Square();

        return result.Value;
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenVector(IDevice device)
    {
        SquareVectorTest<float>(new float[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new float[] { 1, 4, 9 });
        SquareVectorTest<double>(new double[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new double[] { 1, 4, 9 });
        SquareVectorTest<sbyte>(new sbyte[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new sbyte[] { 1, 4, 9 });
        SquareVectorTest<byte>(new byte[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new byte[] { 1, 4, 9 });
        SquareVectorTest<short>(new short[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new short[] { 1, 4, 9 });
        SquareVectorTest<ushort>(new ushort[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new ushort[] { 1, 4, 9 });
        SquareVectorTest<int>(new int[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new int[] { 1, 4, 9 });
        SquareVectorTest<uint>(new uint[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new uint[] { 1, 4, 9 });
        SquareVectorTest<long>(new long[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new long[] { 1, 4, 9 });
        SquareVectorTest<ulong>(new ulong[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new ulong[] { 1, 4, 9 });
    }

    private static Array SquareVectorTest<TNumber>(TNumber[] numbers, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var vector = Tensor.FromArray<TNumber>(numbers).ToVector();

        vector.To(device);

        var result = vector.Square();

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrix(IDevice device)
    {
        SquareMatrixTest<float>(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new float[,] { { 1, 4, 9 }, { 16, 25, 36 } });
        SquareMatrixTest<double>(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new double[,] { { 1, 4, 9 }, { 16, 25, 36 } });
        SquareMatrixTest<sbyte>(new sbyte[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new sbyte[,] { { 1, 4, 9 }, { 16, 25, 36 } });
        SquareMatrixTest<byte>(new byte[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new byte[,] { { 1, 4, 9 }, { 16, 25, 36 } });
        SquareMatrixTest<short>(new short[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new short[,] { { 1, 4, 9 }, { 16, 25, 36 } });
        SquareMatrixTest<ushort>(new ushort[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new ushort[,] { { 1, 4, 9 }, { 16, 25, 36 } });
        SquareMatrixTest<int>(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new int[,] { { 1, 4, 9 }, { 16, 25, 36 } });
        SquareMatrixTest<uint>(new uint[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new uint[,] { { 1, 4, 9 }, { 16, 25, 36 } });
        SquareMatrixTest<long>(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new long[,] { { 1, 4, 9 }, { 16, 25, 36 } });
        SquareMatrixTest<ulong>(new ulong[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new ulong[,] { { 1, 4, 9 }, { 16, 25, 36 } });
    }

    private static Array SquareMatrixTest<TNumber>(TNumber[,] numbers, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var matrix = Tensor.FromArray<TNumber>(numbers).ToMatrix();

        matrix.To(device);

        var result = matrix.Square();

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenTensor(IDevice device)
    {
        SquareTensorTest<float>(new float[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } }, device).Should().BeEquivalentTo(new float[,,] { { { 1, 4, 9 }, { 16, 25, 36 } } });
        SquareTensorTest<double>(new double[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } }, device).Should().BeEquivalentTo(new double[,,] { { { 1, 4, 9 }, { 16, 25, 36 } } });
        SquareTensorTest<sbyte>(new sbyte[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } }, device).Should().BeEquivalentTo(new sbyte[,,] { { { 1, 4, 9 }, { 16, 25, 36 } } });
        SquareTensorTest<byte>(new byte[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } }, device).Should().BeEquivalentTo(new byte[,,] { { { 1, 4, 9 }, { 16, 25, 36 } } });
        SquareTensorTest<short>(new short[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } }, device).Should().BeEquivalentTo(new short[,,] { { { 1, 4, 9 }, { 16, 25, 36 } } });
        SquareTensorTest<ushort>(new ushort[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } }, device).Should().BeEquivalentTo(new ushort[,,] { { { 1, 4, 9 }, { 16, 25, 36 } } });
        SquareTensorTest<int>(new int[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } }, device).Should().BeEquivalentTo(new int[,,] { { { 1, 4, 9 }, { 16, 25, 36 } } });
        SquareTensorTest<uint>(new uint[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } }, device).Should().BeEquivalentTo(new uint[,,] { { { 1, 4, 9 }, { 16, 25, 36 } } });
        SquareTensorTest<long>(new long[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } }, device).Should().BeEquivalentTo(new long[,,] { { { 1, 4, 9 }, { 16, 25, 36 } } });
        SquareTensorTest<ulong>(new ulong[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } }, device).Should().BeEquivalentTo(new ulong[,,] { { { 1, 4, 9 }, { 16, 25, 36 } } });
    }

    private static Array SquareTensorTest<TNumber>(TNumber[,,] numbers, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensor = Tensor.FromArray<TNumber>(numbers);

        tensor.To(device);

        var result = tensor.Square();

        return result.ToArray();
    }
}