// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Integration;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Arithmetic;

public class NegateShould : IntegrationTestBase
{
    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void NegateTensor_GivenScalar(IDevice device)
    {
        NegateScalarTest<BFloat16>(1, device).Should().Be(-1);
        NegateScalarTest<float>(1, device).Should().Be(-1);
        NegateScalarTest<double>(1, device).Should().Be(-1);
        NegateScalarTest<sbyte>(1, device).Should().Be(-1);
        NegateScalarTest<byte>(1, device).Should().Be(1);
        NegateScalarTest<short>(1, device).Should().Be(-1);
        NegateScalarTest<ushort>(1, device).Should().Be(1);
        NegateScalarTest<int>(1, device).Should().Be(-1);
        NegateScalarTest<uint>(1, device).Should().Be(1);
        NegateScalarTest<long>(1, device).Should().Be(-1);
        NegateScalarTest<ulong>(1, device).Should().Be(1);
    }

    private static TNumber NegateScalarTest<TNumber>(TNumber number, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var scalar = new Scalar<TNumber>(number);
        scalar.To(device);

        var result = scalar.Negate();

        return result.Value;
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void NegateTensor_GivenVector(IDevice device)
    {
        NegateVectorTest<BFloat16>(new BFloat16[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new BFloat16[] { -1, -2, -3 });
        NegateVectorTest<float>(new float[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new float[] { -1, -2, -3 });
        NegateVectorTest<double>(new double[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new double[] { -1, -2, -3 });
        NegateVectorTest<sbyte>(new sbyte[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new sbyte[] { -1, -2, -3 });
        NegateVectorTest<byte>(new byte[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new byte[] { 1, 2, 3 });
        NegateVectorTest<short>(new short[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new short[] { -1, -2, -3 });
        NegateVectorTest<ushort>(new ushort[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new ushort[] { 1, 2, 3 });
        NegateVectorTest<int>(new int[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new int[] { -1, -2, -3 });
        NegateVectorTest<uint>(new uint[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new uint[] { 1, 2, 3 });
        NegateVectorTest<long>(new long[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new long[] { -1, -2, -3 });
        NegateVectorTest<ulong>(new ulong[] { 1, 2, 3 }, device).Should().BeEquivalentTo(new ulong[] { 1, 2, 3 });
    }

    private static Array NegateVectorTest<TNumber>(TNumber[] numbers, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var vector = Tensor.FromArray<TNumber>(numbers).ToVector();
        vector.To(device);

        var result = vector.Negate();

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void NegateTensor_GivenMatrix(IDevice device)
    {
        NegateMatrixTest<BFloat16>(new BFloat16[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new BFloat16[,] { { -1, -2, -3 }, { -4, -5, -6 } });
        NegateMatrixTest<float>(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new float[,] { { -1, -2, -3 }, { -4, -5, -6 } });
        NegateMatrixTest<double>(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new double[,] { { -1, -2, -3 }, { -4, -5, -6 } });
        NegateMatrixTest<sbyte>(new sbyte[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new sbyte[,] { { -1, -2, -3 }, { -4, -5, -6 } });
        NegateMatrixTest<byte>(new byte[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new byte[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        NegateMatrixTest<short>(new short[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new short[,] { { -1, -2, -3 }, { -4, -5, -6 } });
        NegateMatrixTest<ushort>(new ushort[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new ushort[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        NegateMatrixTest<int>(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new int[,] { { -1, -2, -3 }, { -4, -5, -6 } });
        NegateMatrixTest<uint>(new uint[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new uint[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        NegateMatrixTest<long>(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new long[,] { { -1, -2, -3 }, { -4, -5, -6 } });
        NegateMatrixTest<ulong>(new ulong[,] { { 1, 2, 3 }, { 4, 5, 6 } }, device).Should().BeEquivalentTo(new ulong[,] { { 1, 2, 3 }, { 4, 5, 6 } });
    }

    private static Array NegateMatrixTest<TNumber>(TNumber[,] numbers, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var matrix = Tensor.FromArray<TNumber>(numbers).ToMatrix();
        matrix.To(device);

        var result = matrix.Negate();

        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void NegateTensor_GivenTensor(IDevice device)
    {
        NegateTensorTest<BFloat16>(new BFloat16[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, device).Should().BeEquivalentTo(new BFloat16[,,] { { { -1, -2, -3 }, { -4, -5, -6 } }, { { -7, -8, -9 }, { -10, -11, -12 } } });
        NegateTensorTest<float>(new float[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, device).Should().BeEquivalentTo(new float[,,] { { { -1, -2, -3 }, { -4, -5, -6 } }, { { -7, -8, -9 }, { -10, -11, -12 } } });
        NegateTensorTest<double>(new double[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, device).Should().BeEquivalentTo(new double[,,] { { { -1, -2, -3 }, { -4, -5, -6 } }, { { -7, -8, -9 }, { -10, -11, -12 } } });
        NegateTensorTest<sbyte>(new sbyte[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, device).Should().BeEquivalentTo(new sbyte[,,] { { { -1, -2, -3 }, { -4, -5, -6 } }, { { -7, -8, -9 }, { -10, -11, -12 } } });
        NegateTensorTest<byte>(new byte[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, device).Should().BeEquivalentTo(new byte[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } });
        NegateTensorTest<short>(new short[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, device).Should().BeEquivalentTo(new short[,,] { { { -1, -2, -3 }, { -4, -5, -6 } }, { { -7, -8, -9 }, { -10, -11, -12 } } });
        NegateTensorTest<ushort>(new ushort[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, device).Should().BeEquivalentTo(new ushort[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } });
        NegateTensorTest<int>(new int[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, device).Should().BeEquivalentTo(new int[,,] { { { -1, -2, -3 }, { -4, -5, -6 } }, { { -7, -8, -9 }, { -10, -11, -12 } } });
        NegateTensorTest<uint>(new uint[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, device).Should().BeEquivalentTo(new uint[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } });
        NegateTensorTest<long>(new long[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, device).Should().BeEquivalentTo(new long[,,] { { { -1, -2, -3 }, { -4, -5, -6 } }, { { -7, -8, -9 }, { -10, -11, -12 } } });
        NegateTensorTest<ulong>(new ulong[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } }, device).Should().BeEquivalentTo(new ulong[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } });
    }

    private static Array NegateTensorTest<TNumber>(TNumber[,,] numbers, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var tensor = Tensor.FromArray<TNumber>(numbers).ToTensor();
        tensor.To(device);

        var result = tensor.Negate();

        return result.ToArray();
    }
}