// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnostics.Windows.Configs;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Common.Numerics.Intrinsics;
using Sci.NET.Common.Numerics.Intrinsics.Extensions;
using Sci.NET.Mathematics.Backends.Managed;

namespace Sci.NET.Mathematics.Benchmarks.SIMD;

[NativeMemoryProfiler]
[MemoryDiagnoser]
[SuppressMessage("Design", "CA1001:Types that own disposable fields should be disposable", Justification = "Benchmark class")]
public class ActivationFunctionBenchmarks
{
    private const int Count = 10000;
    private SystemMemoryBlock<float> _input = null!;
    private SystemMemoryBlock<float> _output = null!;

    [GlobalSetup]
    public void GlobalSetup()
    {
        _input = new SystemMemoryBlock<float>(Count);
        _output = new SystemMemoryBlock<float>(Count);
        _input.Fill(0.5f);
    }

    [Benchmark]
    public void SigmoidScalar()
    {
        _ = LazyParallelExecutor.For(
            0,
            _input.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => _output[i] = 1.0f / (1.0f + float.Exp(-_input[i])));
    }

    [Benchmark]
    public void SigmoidVector()
    {
        var inputMemory = _input;
        var outputMemory = _output;
        var vectorLength = SimdVector.Count<float>();
        var zeroVector = SimdVector.Create(0.0f);
        var oneVector = SimdVector.Create(1.0f);

        var done = LazyParallelExecutor.For(
            0,
            inputMemory.Length - vectorLength,
            ManagedTensorBackend.ParallelizationThreshold,
            vectorLength,
            i =>
            {
                // 1 / 1 + exp(x)
                var vector = inputMemory.UnsafeGetVectorUnchecked<float>(i);
                var negated = zeroVector.Subtract(vector);
                var exp = negated.Exp();
                var onePlusExp = oneVector.Add(exp);
                var divided = oneVector.Divide(onePlusExp);

                outputMemory.UnsafeSetVectorUnchecked(divided, i);
            });

        for (var i = done; i < inputMemory.Length; i++)
        {
            outputMemory[i] = 1.0f / (1.0f + float.Exp(-inputMemory[i]));
        }
    }
}