// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using BenchmarkDotNet.Attributes;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Benchmarks.Managed;

public class ManagedContractionBenchmarks<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    [ParamsSource(nameof(ShapeOptions))]
    public (int Id, Shape LeftShape, Shape RightShape, int[] AxesLeft, int[] AxesRight) Shapes { get; set; } = default!;

    public ICollection<(int Id, Shape LeftShape, Shape RightShape, int[] AxesLeft, int[] AxesRight)> ShapeOptions =>
    [
        (1, new Shape(32, 16), new Shape(16, 64), [1], [0]),
        (2, new Shape(32, 16, 8), new Shape(16, 8, 32), [1, 2], [0, 1]),
        (3, new Shape(8, 16, 32, 64), new Shape(32, 64, 8, 16), [2, 3, 0], [0, 1, 2]),
        (4, new Shape(8, 16, 32, 64, 128), new Shape(32, 64, 128, 16, 8), [2, 3, 4], [0, 1, 2])
    ];

    private Tensor<TNumber> _leftTensor = default!;
    private Tensor<TNumber> _rightTensor = default!;
    private ITensor<TNumber> _result = default!;

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

        _leftTensor = Tensor.Random.Uniform<TNumber>(Shapes.LeftShape, min, max, seed: 123456).ToTensor();
        _rightTensor = Tensor.Random.Uniform<TNumber>(Shapes.RightShape, min, max, seed: 654321).ToTensor();
    }

    [Benchmark]
    public void Contract()
    {
        _result = _leftTensor.Contract(_rightTensor, Shapes.AxesLeft, Shapes.AxesRight);
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        _leftTensor.Dispose();
        _rightTensor.Dispose();
        _result.Dispose();
    }
}