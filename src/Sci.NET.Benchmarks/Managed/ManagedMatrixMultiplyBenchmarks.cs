﻿// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using BenchmarkDotNet.Attributes;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Benchmarks.Managed;

public class ManagedMatrixMultiplyBenchmarks<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    [ParamsSource(nameof(RowsCols))]
    public (int Rows, int Columns) SizeParam { get; set; }

    public ICollection<(int Rows, int Columns)> RowsCols =>
    [
        (100, 200),
        (400, 600),
        (1000, 2000),
        (1024, 1024)
    ];

    private Matrix<TNumber> _leftMatrix = default!;
    private Matrix<TNumber> _rightMatrix = default!;
    private Matrix<TNumber> _result = default!;

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

        _leftMatrix = Tensor.Random.Uniform<TNumber>(new Shape(SizeParam.Rows, SizeParam.Columns), min, max, seed: 123456).ToMatrix();
        _rightMatrix = Tensor.Random.Uniform<TNumber>(new Shape(SizeParam.Columns, SizeParam.Rows), min, max, seed: 654321).ToMatrix();
    }

    [Benchmark]
    public void MatrixMultiply()
    {
        _result = _leftMatrix.MatrixMultiply(_rightMatrix);
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        _leftMatrix.Dispose();
        _rightMatrix.Dispose();
        _result.Dispose();
    }
}