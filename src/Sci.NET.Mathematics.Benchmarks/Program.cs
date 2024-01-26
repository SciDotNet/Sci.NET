// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Running;
using Sci.NET.Mathematics.Benchmarks.SIMD;

BenchmarkRunner.Run<ReduceAddBenchmarks>();

BenchmarkRunner.Run<MatrixMultiplyBenchmarks>();