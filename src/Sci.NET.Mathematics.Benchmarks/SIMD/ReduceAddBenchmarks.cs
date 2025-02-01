// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnostics.Windows.Configs;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Numerics.Intrinsics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Benchmarks.SIMD;

/// <summary>
/// Benchmarks for SIMD reduce add.
/// </summary>
[MemoryDiagnoser]
[NativeMemoryProfiler]
#pragma warning disable CA1515
public class ReduceAddBenchmarks
#pragma warning restore CA1515
{
    private const int N = 1000000;

    private Vector<int> _vector = null!;
    private Vector<int> _result = null!;

    /// <summary>
    /// Setup the vectors.
    /// </summary>
    [GlobalSetup]
    public void Setup()
    {
        _vector = Tensor.Random.Uniform(new Shape(N), 0, 1).ToVector();
        _result = Tensor.Zeros<int>(new Shape(1)).ToVector();
    }

    /// <summary>
    /// Benchmark for reduce add.
    /// </summary>
    [Benchmark]
    public void ReduceAdd()
    {
        var vectorMemoryBlock = (SystemMemoryBlock<int>)_vector.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<int>)_result.Memory;
        var sums = new int[Environment.ProcessorCount];

        vectorMemoryBlock.Fill(0);

        _ = LazyParallelExecutor.For(
            0,
            N,
            1,
            i => sums[Environment.CurrentManagedThreadId % sums.Length] += vectorMemoryBlock[i]);

        resultMemoryBlock[0] = sums.Aggregate(0, (current, partialVectorSum) => current + partialVectorSum);
    }

    /// <summary>
    /// Benchmark for reduce add with SIMD.
    /// </summary>
    [Benchmark]
    public void ReduceAddSimd()
    {
        var vectorMemoryBlock = (SystemMemoryBlock<int>)_vector.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<int>)_result.Memory;
        var vectorLength = System.Numerics.Vector<int>.Count;
        var partialSums = new ConcurrentDictionary<int, ISimdVector<int>>();

        vectorMemoryBlock.Fill(0);

        for (var index = 0; index < partialSums.Count; index++)
        {
            partialSums[index] = SimdVector.Create<int>();
        }

        var done = 0L;

        if (vectorLength <= N)
        {
            done = LazyParallelExecutor.For(
                0,
                N,
                1,
                vectorLength,
                i =>
                {
                    var vector = vectorMemoryBlock.UnsafeGetVectorUnchecked<int>(i);

                    _ = partialSums.AddOrUpdate(
                        Environment.CurrentManagedThreadId,
                        vector,
                        (_, sum) => sum.Add(vector));
                });
        }

        var finalSum = 0;

        for (var i = done * vectorLength; i < N; i++)
        {
            finalSum = vectorMemoryBlock[i];
        }

        foreach (var partialVectorSum in partialSums.Values)
        {
            for (var j = 0; j < vectorLength; j++)
            {
                finalSum += partialVectorSum[j];
            }
        }

        resultMemoryBlock[0] = finalSum;
    }
}