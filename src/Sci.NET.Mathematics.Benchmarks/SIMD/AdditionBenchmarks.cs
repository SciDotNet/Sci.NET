// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnostics.Windows.Configs;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Benchmarks.SIMD;

/// <summary>
/// Benchmarks for SIMD addition.
/// </summary>
[NativeMemoryProfiler]
[MemoryDiagnoser]
#pragma warning disable CA1515
public class AdditionBenchmarks
#pragma warning restore CA1515
{
    private const int N = 50;

    private Tensor<int> _left = null!;
    private Tensor<int> _right = null!;
    private Tensor<int> _result = null!;

    /// <summary>
    /// Setup the tensors.
    /// </summary>
    [GlobalSetup]
    public void Setup()
    {
        _left = Tensor.FromArray<int>(Enumerable.Range(0, N * N * N).ToArray()).Reshape(N, N, N).ToTensor();
        _right = Tensor.FromArray<int>(Enumerable.Range(0, N * N * N).ToArray()).Reshape(N, N, N).ToTensor();
        _result = Tensor.Zeros<int>(new Shape(N, N, N)).ToTensor();
    }

    /// <summary>
    /// Benchmark for addition.
    /// </summary>
    [Benchmark]
    public void Add()
    {
        var leftBlock = (SystemMemoryBlock<int>)_left.Memory;
        var rightBlock = (SystemMemoryBlock<int>)_right.Memory;
        var resultBlock = (SystemMemoryBlock<int>)_result.Memory;

        resultBlock.Fill(0);

        _ = LazyParallelExecutor.For(
            0,
            _result.Shape.ElementCount,
            0,
            i => resultBlock[i] = leftBlock[i] + rightBlock[i]);
    }

    /// <summary>
    /// Benchmark for SIMD addition.
    /// </summary>
    [Benchmark]
    public void AddSimd()
    {
        var leftBlock = (SystemMemoryBlock<int>)_left.Memory;
        var rightBlock = (SystemMemoryBlock<int>)_right.Memory;
        var resultBlock = (SystemMemoryBlock<int>)_result.Memory;
        var vectorSize = System.Numerics.Vector<int>.Count;

        resultBlock.Fill(0);

        var i = LazyParallelExecutor.For(
            0,
            _result.Shape.ElementCount,
            0,
            vectorSize,
            i =>
            {
                var leftVector = leftBlock.UnsafeGetVectorUnchecked<int>(i);
                var rightVector = rightBlock.UnsafeGetVectorUnchecked<int>(i);
                var resultVector = leftVector.Add(rightVector);

                resultBlock.UnsafeSetVectorUnchecked(resultVector, i);
            });

        i *= vectorSize;

        for (; i < _result.Shape.ElementCount; i++)
        {
            resultBlock[i] = leftBlock[i] + rightBlock[i];
        }
    }
}