// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using BenchmarkDotNet.Attributes;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Benchmarks.Managed;

[MaxIterationCount(4096)]
public class ManagedActivationFunctionBenchmarks<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    [ParamsSource(nameof(ShapeOptions))]
    public Shape Shape { get; set; } = default!;

    public ICollection<Shape> ShapeOptions =>
    [
        new Shape(400, 200),
        new Shape(400, 200, 100),
        new Shape(400, 200, 100, 50),
    ];

    private Tensor<TNumber> _tensor = default!;
    private ITensor<TNumber> _result = default!;
    private TNumber _alpha;
    private TNumber _min;
    private TNumber _max;

    [GlobalSetup]
    public void GlobalSetup()
    {
        TNumber min;
        TNumber max;

        if (GenericMath.IsFloatingPoint<TNumber>())
        {
            min = TNumber.CreateChecked(-1f);
            max = TNumber.CreateChecked(1f);
            _alpha = TNumber.CreateChecked(0.01f); // Leaky ReLU alpha
            _min = TNumber.CreateChecked(-1f); // Hard Tanh min
            _max = TNumber.CreateChecked(1f); // Hard Tanh max
        }
        else if (GenericMath.IsSigned<TNumber>())
        {
            min = TNumber.CreateChecked(-10);
            max = TNumber.CreateChecked(10);
            _alpha = TNumber.CreateChecked(1); // Leaky ReLU alpha
            _min = TNumber.CreateChecked(-10); // Hard Tanh min
            _max = TNumber.CreateChecked(10); // Hard Tanh max
        }
        else
        {
            min = TNumber.CreateChecked(1);
            max = TNumber.CreateChecked(10);
            _alpha = TNumber.CreateChecked(0.01f); // Leaky ReLU alpha
            _min = TNumber.CreateChecked(1); // Hard Tanh min
            _max = TNumber.CreateChecked(10); // Hard Tanh max
        }

        _tensor = Tensor.Random.Uniform<TNumber>(Shape, min, max, seed: 123456).ToTensor();
    }

    [Benchmark]
    public void ReLU()
    {
        _result = _tensor.ReLU();
    }

    [Benchmark]
    public void ReLUBackward()
    {
        _result = _tensor.ReLUBackward();
    }

    [Benchmark]
    public void LeakyReLU()
    {
        _result = _tensor.LeakyReLU(_alpha);
    }

    [Benchmark]
    public void LeakyReLUBackward()
    {
        _result = _tensor.LeakyReLUBackward(_alpha);
    }

    [Benchmark]
    public void SoftSign()
    {
        _result = _tensor.SoftSign();
    }

    [Benchmark]
    public void SoftSignBackward()
    {
        _result = _tensor.SoftSignBackward();
    }

    [Benchmark]
    public void HardSigmoid()
    {
        _result = _tensor.HardSigmoid();
    }

    [Benchmark]
    public void HardSigmoidBackward()
    {
        _result = _tensor.HardSigmoidBackward();
    }

    [Benchmark]
    public void HardTanh()
    {
        _result = _tensor.HardTanh(_min, _max);
    }

    [Benchmark]
    public void HardTanhBackward()
    {
        _result = _tensor.HardTanhBackward(_min, _max);
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        _tensor.Dispose();
        _result.Dispose();
    }
}