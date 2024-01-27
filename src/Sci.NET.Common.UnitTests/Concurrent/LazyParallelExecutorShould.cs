// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Concurrency;

namespace Sci.NET.Common.UnitTests.Concurrent;

public class LazyParallelExecutorShould
{
    [Fact]
    public void ExecuteForLoopInParallel()
    {
        const int iterations = 1000000;
        const int parallelizationThreshold = 1000;
        var result = new long[iterations];

        _ = LazyParallelExecutor.For(
            0,
            iterations,
            parallelizationThreshold,
            i => result[i] = i);

        for (var i = 0; i < iterations; i++)
        {
            result[i].Should().Be(i);
        }
    }

    [Fact]
    public void ExecuteForLoopInParallelWithIncrement()
    {
        const int iterations = 1000000;
        const int parallelizationThreshold = 1000;
        const int increment = 2;
        var result = new long[iterations];

        _ = LazyParallelExecutor.For(
            0,
            iterations,
            parallelizationThreshold,
            increment,
            i => result[i] = i);

        for (var i = 0; i < iterations; ++i)
        {
            result[i].Should().Be(i % increment == 0 ? i : 0);
        }
    }
}