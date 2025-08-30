// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using BenchmarkDotNet.Attributes;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Benchmarks.Managed;

public class ManagedBroadcastingBenchmarks<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    [ParamsSource(nameof(ShapeOptions))]
    public (Shape From, Shape To) Shapes { get; set; } = default!;

    public ICollection<(Shape From, Shape To)> ShapeOptions =>
    [
        (new Shape(100, 200), new Shape(100, 100, 200)),
        (new Shape(100, 100, 200), new Shape(50, 100, 100, 200)),
        (new Shape(5000), new Shape(10, 5000)),
    ];

    private Tensor<TNumber> _tensor = default!;
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

        _tensor = Tensor.Random.Uniform<TNumber>(Shapes.From, min, max, seed: 123456).ToTensor();
    }

    [Benchmark]
    public void Broadcast()
    {
        _result = _tensor.Broadcast(Shapes.To);
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        _tensor.Dispose();
        _result.Dispose();
    }
}