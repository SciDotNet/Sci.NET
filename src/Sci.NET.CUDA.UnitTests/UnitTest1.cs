// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.CUDA.Memory;
using Sci.NET.CUDA.Tensors;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.CUDA.UnitTests;

public class UnitTest1
{
    [Fact]
    public void Test1()
    {
        // Allocate 4GB of memory
        var cudaMemory = new CudaMemoryBlock<int>(1024 * 1024 * 1024);
        cudaMemory.Dispose();
    }

    [Fact]
    public void Test2()
    {
        using var left = Tensor
            .FromArray<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 })
            .Reshape(5, 5)
            .ToMatrix();

        using var right = Tensor
            .FromArray<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 })
            .Reshape(5, 5)
            .ToMatrix();

        left.To<CudaComputeDevice>();
        right.To<CudaComputeDevice>();

        var result = left.Dot(right);

        result.To<CpuComputeDevice>();

        var arr = result.ToArray();

        Assert.Equal(25, arr.Length);
    }

    [Fact]
    public void Test3()
    {
        using var left = Tensor
            .FromArray<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 })
            .ToVector();

        using var right = Tensor
            .FromArray<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 })
            .ToVector();

        left.To<CudaComputeDevice>();
        right.To<CudaComputeDevice>();

        var result = left
            .Dot(right)
            .ToScalar();

        result.To<CpuComputeDevice>();

        var arr = result.Value;

        arr
            .Should()
            .BeGreaterThan(0);
    }

    [Fact]
    public void Test4()
    {
        using var left = Tensor
            .FromArray<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 })
            .ToVector();

        left.To<CudaComputeDevice>();

        var result = left.Sin();

        result.To<CpuComputeDevice>();

        var arr = result.ToArray();

        arr
            .Length.Should()
            .NotBe(0);
    }

    [Fact]
    public void Test5()
    {
        // Arrange
        using var left = Tensor
            .FromArray<int>(new int[] { 1, 2, 3, 4 })
            .ToVector();
        using var right = Tensor
            .FromArray<int>(new int[] { 1, 2, 3, 4 })
            .ToTensor();
        var expectedValues = new int[] { 2, 4, 6, 8 };

        left.To<CudaComputeDevice>();
        right.To<CudaComputeDevice>();

        // Act
        var actual = left.Add(right);

        // Assert
        actual
            .ToArray()
            .Should()
            .BeEquivalentTo(expectedValues);
    }
}