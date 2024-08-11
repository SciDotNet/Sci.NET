// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnostics.Windows.Configs;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Benchmarks.Autograd;

[NativeMemoryProfiler]
[MemoryDiagnoser]
[SuppressMessage("Design", "CA1001:Types that own disposable fields should be disposable", Justification = "Benchmark class")]
public class Sin2Benchmarks
{
    private const int CountSmall = 1000;
    private const int CountLarge = 1000000;
    private Tensor<float> _inputSmall = null!;
    private Tensor<float> _outputSmall = null!;
    private Tensor<float> _inputLarge = null!;
    private Tensor<float> _outputLarge = null!;

    [GlobalSetup]
    public void GlobalSetup()
    {
        _inputSmall = new Tensor<float>(new Shape(CountSmall), requiresGradient: true);
        _outputSmall = new Tensor<float>(new Shape(CountSmall), requiresGradient: true);
        _inputSmall.Memory.Fill(0.5f);

        _inputLarge = new Tensor<float>(new Shape(CountLarge), requiresGradient: true);
        _outputLarge = new Tensor<float>(new Shape(CountLarge), requiresGradient: true);
        _inputLarge.Memory.Fill(0.5f);
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        _inputSmall.Dispose();
        _outputSmall.Dispose();
        _inputLarge.Dispose();
        _outputLarge.Dispose();
    }

    [Benchmark]
    public void Sin2BackwardMultipleOperationsSmall()
    {
        using var multiplier = new Scalar<float>(2.0f);
        using var sinX = _inputSmall.Sin();
        using var cosX = _inputSmall.Cos();

        _outputSmall = multiplier * sinX * cosX;
    }

    [Benchmark]
    public void Sin2BackwardSpecificKernelSmall()
    {
        _outputSmall = TensorServiceProvider.GetTensorOperationServiceProvider().GetTrigonometryService().Sin2Backwards(_inputSmall).ToTensor();
    }

    [Benchmark]
    public void Sin2BackwardMultipleOperationsLarge()
    {
        using var multiplier = new Scalar<float>(2.0f);
        using var sinX = _inputLarge.Sin();
        using var cosX = _inputLarge.Cos();

        _outputLarge = multiplier * sinX * cosX;
    }

    [Benchmark]
    public void Sin2BackwardSpecificKernelLarge()
    {
        _outputLarge = TensorServiceProvider.GetTensorOperationServiceProvider().GetTrigonometryService().Sin2Backwards(_inputLarge).ToTensor();
    }
}