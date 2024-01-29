// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnostics.Windows.Configs;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Benchmarks.SIMD;

/// <summary>
/// Benchmarks for SIMD matrix multiplication.
/// </summary>
[MemoryDiagnoser]
[NativeMemoryProfiler]
public class MatrixMultiplyBenchmarks
{
    private const int N = 512;
    private const int M = 768;
    private const int P = 512;

    private Matrix<float> _left = null!;
    private Matrix<float> _right = null!;
    private Matrix<float> _result = null!;

    /// <summary>
    /// Setup the matrices.
    /// </summary>
    [GlobalSetup]
    public void Setup()
    {
        _left = Tensor.Random.Uniform<float>(new Shape(M, N), 0, 1).ToMatrix();
        _right = Tensor.Random.Uniform<float>(new Shape(N, P), 0, 1).ToMatrix();
        _result = Tensor.Zeros<float>(new Shape(M, P)).ToMatrix();
    }

    /// <summary>
    /// Benchmark for matrix multiplication.
    /// </summary>
    [Benchmark]
    public void MatrixMultiply()
    {
        var leftMemoryBlock = (SystemMemoryBlock<float>)_left.Memory;
        var rightMemoryBlock = (SystemMemoryBlock<float>)_right.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<float>)_result.Memory;

        resultMemoryBlock.Fill(0);

        _ = LazyParallelExecutor.For(
            0,
            _result.Rows,
            0,
            _result.Columns,
            1000,
            (i, j) =>
            {
                var sum = 0.0f;

                for (var k = 0; k < _left.Columns; k++)
                {
                    sum += leftMemoryBlock[(i * _left.Columns) + k] *
                           rightMemoryBlock[(k * _right.Columns) + j];
                }

                resultMemoryBlock[(i * _result.Columns) + j] = sum;
            });
    }

    /// <summary>
    /// Benchmark for matrix multiplication with SIMD.
    /// </summary>
    [Benchmark]
    public void MatrixMultiplySimd()
    {
        var leftMemoryBlock = (SystemMemoryBlock<float>)_left.Memory;
        var rightMemoryBlock = (SystemMemoryBlock<float>)_right.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<float>)_result.Memory;
        var vectorSize = System.Numerics.Vector<float>.Count;
        var kLength = _left.Columns;
        var jLength = _result.Columns;

        resultMemoryBlock.Fill(0);

        _ = LazyParallelExecutor.For(
            0,
            _result.Rows,
            0,
            _result.Columns,
            1000,
            (i, j) =>
            {
                var sum = 0.0f;
                var k = 0;

                for (; k <= kLength - vectorSize; k += vectorSize)
                {
                    var leftSpan = leftMemoryBlock.AsSpan((i * kLength) + k, vectorSize);
                    var rightSpan = rightMemoryBlock.AsSpan((k * jLength) + j, vectorSize);
                    var vLeft = new System.Numerics.Vector<float>(leftSpan);
                    var vRight = new System.Numerics.Vector<float>(rightSpan);

                    sum += Vector.Dot(vLeft, vRight);
                }

                for (; k < kLength; k++)
                {
                    sum += leftMemoryBlock[(i * kLength) + k] * rightMemoryBlock[(k * jLength) + j];
                }

                resultMemoryBlock[(i * jLength) + j] = sum;
            });
    }
}