// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.UnitTests.Tensors.Arithmetic;

public class ArithmeticAddKernelShould
{
    [Fact]
    public void ReturnExpectedResult_GivenFloatScalars()
    {
        // Arrange
        var left = new Scalar<float>(1);
        var right = new Scalar<float>(2);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        result.Value.Should().Be(3);
    }

    [Fact]
    public void ReturnExpectedResult_GivenDoubleScalars()
    {
        // Arrange
        var left = new Scalar<double>(1);
        var right = new Scalar<double>(2);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        result.Value.Should().Be(3);
    }

    [Fact]
    public void ReturnExpectedResult_GivenUint8Scalars()
    {
        // Arrange
        var left = new Scalar<byte>(1);
        var right = new Scalar<byte>(2);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        result.Value.Should().Be(3);
    }

    [Fact]
    public void ReturnExpectedResult_GivenInt8Scalars()
    {
        // Arrange
        var left = new Scalar<sbyte>(1);
        var right = new Scalar<sbyte>(2);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        result.Value.Should().Be(3);
    }

    [Fact]
    public void ReturnExpectedResult_GivenUint16Scalars()
    {
        // Arrange
        var left = new Scalar<ushort>(1);
        var right = new Scalar<ushort>(2);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        result.Value.Should().Be(3);
    }

    [Fact]
    public void ReturnExpectedResult_GivenInt16Scalars()
    {
        // Arrange
        var left = new Scalar<short>(1);
        var right = new Scalar<short>(2);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        result.Value.Should().Be(3);
    }

    [Fact]
    public void ReturnExpectedResult_GivenUint32Scalars()
    {
        // Arrange
        var left = new Scalar<uint>(1);
        var right = new Scalar<uint>(2);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        result.Value.Should().Be(3);
    }

    [Fact]
    public void ReturnExpectedResult_GivenInt32Scalars()
    {
        // Arrange
        var left = new Scalar<int>(1);
        var right = new Scalar<int>(2);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        result.Value.Should().Be(3);
    }

    [Fact]
    public void ReturnExpectedResult_GivenUint64Scalars()
    {
        // Arrange
        var left = new Scalar<ulong>(1);
        var right = new Scalar<ulong>(2);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        result.Value.Should().Be(3);
    }

    [Fact]
    public void ReturnExpectedResult_GivenInt64Scalars()
    {
        // Arrange
        var left = new Scalar<long>(1);
        var right = new Scalar<long>(2);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        result.Value.Should().Be(3);
    }

    [Fact]
    public void ReturnExpectedResult_GivenFloatScalarAndVector()
    {
        // Arrange
        var left = new Scalar<float>(1);
        var right = Tensor.FromArray<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new float[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnExpectedResult_GivenDoubleScalarAndVector()
    {
        // Arrange
        var left = new Scalar<double>(1);
        var right = Tensor.FromArray<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new double[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnExpectedResult_GivenUint8ScalarAndVector()
    {
        // Arrange
        var left = new Scalar<byte>(1);
        var right = Tensor.FromArray<byte>(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new byte[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnExpectedResult_GivenInt8ScalarAndVector()
    {
        // Arrange
        var left = new Scalar<sbyte>(1);
        var right = Tensor.FromArray<sbyte>(new sbyte[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new sbyte[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnExpectedResult_GivenUint16ScalarAndVector()
    {
        // Arrange
        var left = new Scalar<ushort>(1);
        var right = Tensor.FromArray<ushort>(new ushort[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new ushort[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnExpectedResult_GivenInt16ScalarAndVector()
    {
        // Arrange
        var left = new Scalar<short>(1);
        var right = Tensor.FromArray<short>(new short[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new short[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnExpectedResult_GivenUint32ScalarAndVector()
    {
        // Arrange
        var left = new Scalar<uint>(1);
        var right = Tensor.FromArray<uint>(new uint[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new uint[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnExpectedResult_GivenInt32ScalarAndVector()
    {
        // Arrange
        var left = new Scalar<int>(1);
        var right = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new int[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnExpectedResult_GivenUint64ScalarAndVector()
    {
        // Arrange
        var left = new Scalar<ulong>(1);
        var right = Tensor.FromArray<ulong>(new ulong[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new ulong[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnExpectedResult_GivenInt64ScalarAndVector()
    {
        // Arrange
        var left = new Scalar<long>(1);
        var right = Tensor.FromArray<long>(new long[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new long[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenFloatScalarAndMatrix()
    {
        // Arrange
        var left = new Scalar<float>(1);
        var right = Tensor.FromArray<float>(new float[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }).ToMatrix();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new float[,] { { 2, 3, 4, 5 }, { 6, 7, 8, 9 } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenDoubleScalarAndMatrix()
    {
        // Arrange
        var left = new Scalar<double>(1);
        var right = Tensor.FromArray<double>(new double[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }).ToMatrix();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new double[,] { { 2, 3, 4, 5 }, { 6, 7, 8, 9 } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenUint8ScalarAndMatrix()
    {
        // Arrange
        var left = new Scalar<byte>(1);
        var right = Tensor.FromArray<byte>(new byte[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }).ToMatrix();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new byte[,] { { 2, 3, 4, 5 }, { 6, 7, 8, 9 } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenInt8ScalarAndMatrix()
    {
        // Arrange
        var left = new Scalar<sbyte>(1);
        var right = Tensor.FromArray<sbyte>(new sbyte[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }).ToMatrix();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new sbyte[,] { { 2, 3, 4, 5 }, { 6, 7, 8, 9 } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenUint16ScalarAndMatrix()
    {
        // Arrange
        var left = new Scalar<ushort>(1);
        var right = Tensor.FromArray<ushort>(new ushort[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }).ToMatrix();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new ushort[,] { { 2, 3, 4, 5 }, { 6, 7, 8, 9 } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenInt16ScalarAndMatrix()
    {
        // Arrange
        var left = new Scalar<short>(1);
        var right = Tensor.FromArray<short>(new short[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }).ToMatrix();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new short[,] { { 2, 3, 4, 5 }, { 6, 7, 8, 9 } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenUint32ScalarAndMatrix()
    {
        // Arrange
        var left = new Scalar<uint>(1);
        var right = Tensor.FromArray<uint>(new uint[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }).ToMatrix();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new uint[,] { { 2, 3, 4, 5 }, { 6, 7, 8, 9 } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenInt32ScalarAndMatrix()
    {
        // Arrange
        var left = new Scalar<int>(1);
        var right = Tensor.FromArray<int>(new int[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }).ToMatrix();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new int[,] { { 2, 3, 4, 5 }, { 6, 7, 8, 9 } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenUint64ScalarAndMatrix()
    {
        // Arrange
        var left = new Scalar<ulong>(1);
        var right = Tensor.FromArray<ulong>(new ulong[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }).ToMatrix();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new ulong[,] { { 2, 3, 4, 5 }, { 6, 7, 8, 9 } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenInt64ScalarAndMatrix()
    {
        // Arrange
        var left = new Scalar<long>(1);
        var right = Tensor.FromArray<long>(new long[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }).ToMatrix();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        result.To<CpuComputeDevice>();
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new long[,] { { 2, 3, 4, 5 }, { 6, 7, 8, 9 } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenFloatScalarAndTensor()
    {
        // Arrange
        var left = new Scalar<float>(1);
        var right = Tensor.FromArray<float>(new float[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }).ToTensor();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new float[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenDoubleScalarAndTensor()
    {
        // Arrange
        var left = new Scalar<double>(1);
        var right = Tensor.FromArray<double>(new double[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }).ToTensor();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new double[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenUint8ScalarAndTensor()
    {
        // Arrange
        var left = new Scalar<byte>(1);
        var right = Tensor.FromArray<byte>(new byte[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }).ToTensor();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new byte[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenInt8ScalarAndTensor()
    {
        // Arrange
        var left = new Scalar<sbyte>(1);
        var right = Tensor.FromArray<sbyte>(new sbyte[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }).ToTensor();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new sbyte[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenUint16ScalarAndTensor()
    {
        // Arrange
        var left = new Scalar<ushort>(1);
        var right = Tensor.FromArray<ushort>(new ushort[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }).ToTensor();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new ushort[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenInt16ScalarAndTensor()
    {
        // Arrange
        var left = new Scalar<short>(1);
        var right = Tensor.FromArray<short>(new short[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }).ToTensor();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new short[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenUint32ScalarAndTensor()
    {
        // Arrange
        var left = new Scalar<uint>(1);
        var right = Tensor.FromArray<uint>(new uint[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }).ToTensor();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new uint[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenInt32ScalarAndTensor()
    {
        // Arrange
        var left = new Scalar<int>(1);
        var right = Tensor.FromArray<int>(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }).ToTensor();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new int[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenUint64ScalarAndTensor()
    {
        // Arrange
        var left = new Scalar<ulong>(1);
        var right = Tensor.FromArray<ulong>(new ulong[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }).ToTensor();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new ulong[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenInt64ScalarAndTensor()
    {
        // Arrange
        var left = new Scalar<long>(1);
        var right = Tensor.FromArray<long>(new long[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }).ToTensor();
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new long[,,] { { { 2, 3 }, { 4, 5 } }, { { 6, 7 }, { 8, 9 } } });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenFloatVectorAndScalar()
    {
        // Arrange
        var left = Tensor.FromArray<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        var right = new Scalar<float>(1);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new float[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenDoubleVectorAndScalar()
    {
        // Arrange
        var left = Tensor.FromArray<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        var right = new Scalar<double>(1);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new double[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenUint8VectorAndScalar()
    {
        // Arrange
        var left = Tensor.FromArray<byte>(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        var right = new Scalar<byte>(1);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new byte[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenInt8VectorAndScalar()
    {
        // Arrange
        var left = Tensor.FromArray<sbyte>(new sbyte[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        var right = new Scalar<sbyte>(1);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new sbyte[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenUint16VectorAndScalar()
    {
        // Arrange
        var left = Tensor.FromArray<ushort>(new ushort[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        var right = new Scalar<ushort>(1);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new ushort[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenInt16VectorAndScalar()
    {
        // Arrange
        var left = Tensor.FromArray<short>(new short[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        var right = new Scalar<short>(1);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new short[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenUint32VectorAndScalar()
    {
        // Arrange
        var left = Tensor.FromArray<uint>(new uint[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        var right = new Scalar<uint>(1);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new uint[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenInt32VectorAndScalar()
    {
        // Arrange
        var left = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        var right = new Scalar<int>(1);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new int[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }

    [Fact]
    public void ReturnsExpectedResult_GivenUint64VectorAndScalar()
    {
        // Arrange
        var left = Tensor.FromArray<ulong>(new ulong[] { 1, 2, 3, 4, 5, 6, 7, 8 }).ToVector();
        var right = new Scalar<ulong>(1);
        left.To<CpuComputeDevice>();
        right.To<CpuComputeDevice>();

        // Act
        var result = left.Add(right);

        // Assert
        var arr = result.ToArray();

        arr
            .Should()
            .BeEquivalentTo(new ulong[] { 2, 3, 4, 5, 6, 7, 8, 9 });
    }
}