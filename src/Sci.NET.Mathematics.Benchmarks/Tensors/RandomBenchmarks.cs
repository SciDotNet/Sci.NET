// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using BenchmarkDotNet.Attributes;
using Sci.NET.CUDA.Tensors;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Benchmarks.Tensors;

/// <summary>
/// Benchmarks for random number generation.
/// </summary>
[SuppressMessage("Performance", "CA1822:Mark members as static", Justification = "BenchmarkDotNet requires instance methods.")]
public class RandomBenchmarks
{
    /// <summary>
    /// Benchmarks the generation of random numbers from a normal distribution on the CPU.
    /// </summary>
    [Benchmark]
    public void UniformManagedCpu()
    {
        using var tensor = Tensor.Random.Uniform<float, CpuComputeDevice>(
            new Shape(500, 200, 400),
            0,
            1,
            4);
    }

    /// <summary>
    /// Benchmarks the generation of random numbers from a normal distribution on the GPU using CUDA.
    /// </summary>
    [Benchmark]
    public void UniformCUDA()
    {
        using var tensor = Tensor.Random.Uniform<float, CudaComputeDevice>(
            new Shape(500, 200, 400),
            0,
            1,
            4);
    }
}