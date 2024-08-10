// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using BenchmarkDotNet.Running;
using Sci.NET.Mathematics.Benchmarks.Autograd;

[assembly:ExcludeFromCodeCoverage]

BenchmarkRunner.Run<Sin2Benchmarks>();

// BenchmarkRunner.Run<AdditionBenchmarks>();
// BenchmarkRunner.Run<ReduceAddBenchmarks>();
// BenchmarkRunner.Run<MatrixMultiplyBenchmarks>();