// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using BenchmarkDotNet.Attributes;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Benchmarks.Managed;

public class ManagedEqualityBenchmarks<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    [ParamsSource(nameof(ShapeOptions))]
    public Shape Shape { get; set; } = default!;

    public ICollection<Shape> ShapeOptions =>
    [
        new Shape(400, 200),
        new Shape(400, 200, 100),
        new Shape(400, 200, 100, 50)
    ];

    private Tensor<TNumber> _leftTensor = default!;
    private Tensor<TNumber> _rightTensor = default!;
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

        _leftTensor = Tensor.Random.Uniform<TNumber>(Shape, min, max, seed: 123456).ToTensor();
        _rightTensor = Tensor.Random.Uniform<TNumber>(Shape, min, max, seed: 654321).ToTensor();
    }

    [Benchmark]
    public void Equals()
    {
        _result = _leftTensor.PointwiseEquals(_rightTensor);
    }

    [Benchmark]
    public void NotEquals()
    {
        _result = _leftTensor.PointwiseNotEquals(_rightTensor);
    }

    [Benchmark]
    public void GreaterThan()
    {
        _result = _leftTensor.PointwiseGreaterThan(_rightTensor);
    }

    [Benchmark]
    public void GreaterThanOrEquals()
    {
        _result = _leftTensor.PointwiseGreaterThanOrEqual(_rightTensor);
    }

    [Benchmark]
    public void LessThan()
    {
        _result = _leftTensor.PointwiseLessThan(_rightTensor);
    }

    [Benchmark]
    public void LessThanOrEquals()
    {
        _result = _leftTensor.PointwiseLessThanOrEqual(_rightTensor);
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        _leftTensor.Dispose();
        _rightTensor.Dispose();
        _result.Dispose();
    }
}