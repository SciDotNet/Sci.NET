// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.Concurrency;

/// <summary>
/// A helper class for executing loops in parallel.
/// </summary>
[PublicAPI]
public static class LazyParallelExecutor
{
    /// <summary>
    /// Executes a loop in parallel if the number of iterations is greater than the threshold.
    /// </summary>
    /// <param name="fromInclusive">The index to start from (inclusive).</param>
    /// <param name="toExclusive">The index to iterate to (exclusive).</param>
    /// <param name="parallelizationThreshold">The threshold before the loop is executed in parallel.</param>
    /// <param name="body">The body of the for loop.</param>
    public static void For(long fromInclusive, long toExclusive, long parallelizationThreshold, Action<long> body)
    {
        if (toExclusive - fromInclusive < parallelizationThreshold)
        {
            for (var i = fromInclusive; i < toExclusive; i++)
            {
                body(i);
            }
        }
        else
        {
            _ = Parallel.For(fromInclusive, toExclusive, body);
        }
    }

    /// <summary>
    /// Executes a loop in parallel if the number of iterations is greater than the threshold.
    /// </summary>
    /// <param name="iFromInclusive">The index to start from (inclusive) for the outer loop.</param>
    /// <param name="iToExclusive">The index to iterate to (exclusive) for outer loop.</param>
    /// <param name="jFromInclusive">The index to start from (inclusive) for the inner loop.</param>
    /// <param name="jToExclusive">The index to iterate to (exclusive) for inner loop.</param>
    /// <param name="parallelizationThreshold">The threshold before the loop is executed in parallel.</param>
    /// <param name="body">The body of the for loop.</param>
    public static void For(
        long iFromInclusive,
        long iToExclusive,
        long jFromInclusive,
        long jToExclusive,
        long parallelizationThreshold,
        Action<long, long> body)
    {
        var loopIterations = (iToExclusive - iFromInclusive) * (jToExclusive - jFromInclusive);

        if (loopIterations < parallelizationThreshold)
        {
            for (var i = iFromInclusive; i < iToExclusive; i++)
            {
                for (var j = jFromInclusive; j < jToExclusive; j++)
                {
                    body(i, j);
                }
            }
        }
        else
        {
            var partitioner = new TwoDimensionalPartitioner(iFromInclusive, iToExclusive, jFromInclusive, jToExclusive);
            _ = Parallel.ForEach(partitioner, pair => body(pair.Item1, pair.Item2));
        }
    }
}