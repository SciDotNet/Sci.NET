// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Loggers;
using BenchmarkDotNet.Toolchains.InProcess.NoEmit;

namespace Sci.NET.Mathematics.Benchmarks.Config;

/// <summary>
/// Configures the benchmarking environment to be friendly to anti-virus software.
/// </summary>
public class AntiVirusFriendlyConfig : ManualConfig
{
    /// <summary>
    /// Initializes a new instance of the <see cref="AntiVirusFriendlyConfig"/> class.
    /// </summary>
    public AntiVirusFriendlyConfig()
    {
        _ = AddJob(
                Job.ShortRun
                    .WithToolchain(InProcessNoEmitToolchain.Instance))
            .AddColumnProvider(DefaultColumnProviders.Instance)
            .AddDiagnoser(new MemoryDiagnoser(new MemoryDiagnoserConfig()))
            .AddLogger(new ConsoleLogger())
            .AddExporter(new HtmlExporter());
    }
}