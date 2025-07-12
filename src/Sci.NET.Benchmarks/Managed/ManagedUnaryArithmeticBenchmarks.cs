// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using BenchmarkDotNet.Attributes;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Benchmarks.Managed;

public class ManagedUnaryArithmeticBenchmarks<TNumber>
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
    private Tensor<TNumber> _result = default!;

    [GlobalSetup]
    public void GlobalSetup()
    {
        TNumber min;
        TNumber max;

        if (GenericMath.IsFloatingPoint<TNumber>())
        {
            min = TNumber.CreateChecked(-1f);
            max = TNumber.CreateChecked(1f);
        }
        else if (GenericMath.IsSigned<TNumber>())
        {
            min = TNumber.CreateChecked(-10);
            max = TNumber.CreateChecked(10);
        }
        else
        {
            min = TNumber.CreateChecked(1);
            max = TNumber.CreateChecked(10);
        }

        _tensor = Tensor.Random.Uniform<TNumber>(Shape, min, max, seed: 123456).ToTensor();
    }

    [Benchmark]
    public void Abs()
    {
        _result = _tensor.Abs();
    }

    [Benchmark]
    public void Sqrt()
    {
        _result = _tensor.Sqrt();
    }

    [Benchmark]
    public void Negate()
    {
        _result = _tensor.Negate();
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        _tensor.Dispose();
        _result.Dispose();
    }
}