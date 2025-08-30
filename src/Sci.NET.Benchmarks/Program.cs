// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Running;
using Sci.NET.Benchmarks.Managed;

// Linear algebra benchmarks
BenchmarkRunner.Run<ManagedMatrixMultiplyBenchmarks<float>>();
BenchmarkRunner.Run<ManagedMatrixMultiplyBenchmarks<double>>();
BenchmarkRunner.Run<ManagedInnerProductBenchmarks<float>>();
BenchmarkRunner.Run<ManagedInnerProductBenchmarks<double>>();
BenchmarkRunner.Run<ManagedContractionBenchmarks<float>>();
BenchmarkRunner.Run<ManagedContractionBenchmarks<double>>();

// Arithmetic benchmarks
BenchmarkRunner.Run<ManagedBinaryArithmeticBenchmarks<float>>();
BenchmarkRunner.Run<ManagedBinaryArithmeticBenchmarks<double>>();
BenchmarkRunner.Run<ManagedUnaryArithmeticBenchmarks<float>>();
BenchmarkRunner.Run<ManagedUnaryArithmeticBenchmarks<double>>();

// Activation function benchmarks
BenchmarkRunner.Run<ManagedActivationFunctionBenchmarks<float>>();
BenchmarkRunner.Run<ManagedActivationFunctionBenchmarks<double>>();

// Broadcasting benchmarks
BenchmarkRunner.Run<ManagedBroadcastingBenchmarks<float>>();
BenchmarkRunner.Run<ManagedBroadcastingBenchmarks<double>>();

// Equality benchmarks
BenchmarkRunner.Run<ManagedEqualityBenchmarks<float>>();
BenchmarkRunner.Run<ManagedEqualityBenchmarks<double>>();

// Reshape benchmarks
BenchmarkRunner.Run<ManagedPermutationBenchmarks<float>>();
BenchmarkRunner.Run<ManagedPermutationBenchmarks<double>>();