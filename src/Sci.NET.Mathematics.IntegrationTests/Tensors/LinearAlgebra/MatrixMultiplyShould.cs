// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.LinearAlgebra;

public class MatrixMultiplyShould : IntegrationTestBase
{
    private static Array MatrixMatrixTest<TNumber>(TNumber[,] left, TNumber[,] right, IDevice device)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftScalar = Tensor.FromArray<TNumber>(left).ToMatrix();
        var rightVector = Tensor.FromArray<TNumber>(right).ToMatrix();
        leftScalar.To(device);
        rightVector.To(device);

        var result = leftScalar.MatrixMultiply(rightVector);

        result.To<CpuComputeDevice>();
        return result.ToArray();
    }

    [Theory]
    [MemberData(nameof(ComputeDevices))]
    public void ReturnExpectedResult_GivenMatrixAndMatrix(IDevice device)
    {
        MatrixMatrixTest<float>(
                new float[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new float[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new float[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<double>(
                new double[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new double[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new double[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<byte>(
                new byte[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new byte[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new byte[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<sbyte>(
                new sbyte[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new sbyte[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new sbyte[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<ushort>(
                new ushort[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new ushort[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new ushort[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<short>(
                new short[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new short[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new short[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<uint>(
                new uint[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new uint[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new uint[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<int>(
                new int[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new int[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new int[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<ulong>(
                new ulong[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new ulong[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new ulong[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<long>(
                new long[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new long[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new long[,] { { 30, 30 }, { 30, 30 } });

        MatrixMatrixTest<BFloat16>(
                new BFloat16[,] { { 1, 2, 3, 4 }, { 1, 2, 3, 4 } },
                new BFloat16[,] { { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 } },
                device)
            .Should()
            .BeEquivalentTo(new BFloat16[,] { { 30, 30 }, { 30, 30 } });
    }
}