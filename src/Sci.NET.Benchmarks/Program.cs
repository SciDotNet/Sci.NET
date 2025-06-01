// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Running;
using Sci.NET.Benchmarks.Managed.LinearAlgebra;

BenchmarkRunner.Run<ManagedMatrixMultiplyBenchmarks<float>>();
BenchmarkRunner.Run<ManagedMatrixMultiplyBenchmarks<double>>();
BenchmarkRunner.Run<ManagedInnerProductBenchmarks<float>>();
BenchmarkRunner.Run<ManagedInnerProductBenchmarks<double>>();