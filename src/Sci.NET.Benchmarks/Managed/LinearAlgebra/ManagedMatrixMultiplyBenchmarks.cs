// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using BenchmarkDotNet.Attributes;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Benchmarks.Managed.LinearAlgebra;

[SuppressMessage("Design", "CA1051:Do not declare visible instance fields", Justification = "Required for BenchmarkDotNet to work correctly.")]
[SuppressMessage("StyleCop.CSharp.MaintainabilityRules", "SA1401:Fields should be private", Justification = "Required for BenchmarkDotNet to work correctly.")]
[SuppressMessage("Roslynator", "RCS1085:Use auto-implemented property", Justification = "Required for BenchmarkDotNet to work correctly.")]
[SuppressMessage("Roslynator", "RCS1158:Static member in generic type should use a type parameter", Justification = "Required for BenchmarkDotNet to work correctly.")]
[SuppressMessage("Design", "CA1000:Do not declare static members on generic types", Justification = "Required for BenchmarkDotNet to work correctly.")]
public class ManagedMatrixMultiplyBenchmarks<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    [ParamsSource(nameof(RowsCols))]
    public (int Rows, int Columns) SizeParam
    {
        get => _size;
        set => _size = value;
    }

    public static IEnumerable<(int Rows, int Columns)> RowsCols => new[]
    {
        (100, 200),
        (400, 600),
        (1000, 2000),
        (1024, 1024)
    };

    private (int Rows, int Columns) _size;

    private Matrix<TNumber> _leftMatrix = default!;

    private Matrix<TNumber> _rightMatrix = default!;

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

        _leftMatrix = Tensor.Random.Uniform<TNumber>(new Shape(_size.Rows, _size.Columns), min, max, seed: 123456).ToMatrix();
        _rightMatrix = Tensor.Random.Uniform<TNumber>(new Shape(_size.Columns, _size.Rows), min, max, seed: 654321).ToMatrix();
    }

    [Benchmark]
    public void MatrixMultiply()
    {
        var result = _leftMatrix.MatrixMultiply(_rightMatrix);

        result.Dispose();
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        _leftMatrix.Dispose();
        _rightMatrix.Dispose();
    }
}