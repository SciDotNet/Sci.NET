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
    /// <returns>The number of iterations executed.</returns>
    public static long For(long fromInclusive, long toExclusive, long parallelizationThreshold, Action<long> body)
    {
        return For(
            fromInclusive,
            toExclusive,
            parallelizationThreshold,
            1,
            body);
    }

    /// <summary>
    /// Executes a loop in parallel if the number of iterations is greater than the threshold.
    /// </summary>
    /// <param name="fromInclusive">The index to start from (inclusive).</param>
    /// <param name="toExclusive">The index to iterate to (exclusive).</param>
    /// <param name="parallelizationThreshold">The threshold before the loop is executed in parallel.</param>
    /// <param name="increment">The increment to use for the loop.</param>
    /// <param name="body">The body of the for loop.</param>
    /// <returns>The number of iterations executed.</returns>
    public static long For(
        long fromInclusive,
        long toExclusive,
        long parallelizationThreshold,
        long increment,
        Action<long> body)
    {
        if (toExclusive - fromInclusive < parallelizationThreshold)
        {
            var i = fromInclusive;

            for (; i < toExclusive; i += increment)
            {
                body(i);
            }

            return i;
        }

        _ = Parallel.For(
            fromInclusive,
            (toExclusive - fromInclusive) / increment,
            i =>
            {
                var idx = fromInclusive + (i * increment);
                body(idx);
            });

        return (toExclusive - fromInclusive + increment - 1) / increment;
    }

    /// <summary>
    /// Executes a loop in parallel if the number of iterations is greater than the threshold.
    /// </summary>
    /// <param name="iFromInclusive">The index to start from (inclusive) for the outer loop.</param>
    /// <param name="iToExclusive">The index to iterate to (exclusive) for outer loop.</param>
    /// <param name="jFromInclusive">The index to start from (inclusive) for the inner loop.</param>
    /// <param name="jToExclusive">The index to iterate to (exclusive) for inner loop.</param>
    /// <param name="parallelizationThreshold">The threshold before the loop is executed in parallel.</param>
    /// <param name="iIncrement">The increment to use for the outer loop.</param>
    /// <param name="jIncrement">The increment to use for the inner loop.</param>
    /// <param name="body">The body of the for loop.</param>
    /// <returns>The number of iterations executed.</returns>
    public static (long ICount, long JCount) For(
        long iFromInclusive,
        long iToExclusive,
        long jFromInclusive,
        long jToExclusive,
        long parallelizationThreshold,
        long iIncrement,
        long jIncrement,
        Action<long, long> body)
    {
        var loopIterations = (iToExclusive - iFromInclusive) * (jToExclusive - jFromInclusive);
        var iRange = iToExclusive - iFromInclusive;
        var jRange = jToExclusive - jFromInclusive;

        if (loopIterations < parallelizationThreshold)
        {
            for (var i = iFromInclusive; i < iToExclusive; i += iIncrement)
            {
                for (var j = jFromInclusive; j < jToExclusive; j += jIncrement)
                {
                    body(i, j);
                }
            }
        }
        else
        {
            var shouldParallelizeOuter = iRange >= jRange;

            _ = Parallel.For(
                iFromInclusive,
                iToExclusive,
                new ParallelOptions { MaxDegreeOfParallelism = shouldParallelizeOuter ? Environment.ProcessorCount : 1 },
                i =>
                {
                    _ = Parallel.For(
                        jFromInclusive,
                        jToExclusive,
                        new ParallelOptions { MaxDegreeOfParallelism = shouldParallelizeOuter ? 1 : Environment.ProcessorCount },
                        jj => body(i, jj));
                });
        }

        var iCount = (iToExclusive - iFromInclusive + iIncrement - 1) / iIncrement;
        var jCount = (jToExclusive - jFromInclusive + jIncrement - 1) / jIncrement;

        return (iCount, jCount);
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
    /// <returns>The number of iterations executed.</returns>
    public static (long ICount, long JCount) For(
        long iFromInclusive,
        long iToExclusive,
        long jFromInclusive,
        long jToExclusive,
        long parallelizationThreshold,
        Action<long, long> body)
    {
        return For(
            iFromInclusive,
            iToExclusive,
            jFromInclusive,
            jToExclusive,
            parallelizationThreshold,
            1,
            1,
            body);
    }

    /// <summary>
    /// Executes a loop in parallel over blocks of indices.
    /// </summary>
    /// <param name="iFrom">The index to start from (inclusive) for the outer loop.</param>
    /// <param name="iTo">The index to iterate to (exclusive) for outer loop.</param>
    /// <param name="jFrom">The index to start from (inclusive) for the inner loop.</param>
    /// <param name="jTo">The index to iterate to (exclusive) for inner loop.</param>
    /// <param name="iBlock">The size of the block for the outer loop.</param>
    /// <param name="jBlock">The size of the block for the inner loop.</param>
    /// <param name="body">The body of the for loop.</param>
    public static void ForBlocked(
        long iFrom,
        long iTo,
        long jFrom,
        long jTo,
        int iBlock,
        int jBlock,
        Action<long, long> body)
    {
        var iTiles = (iTo - iFrom + iBlock - 1) / iBlock;
        var jTiles = (jTo - jFrom + jBlock - 1) / jBlock;
        var tileCount = iTiles * jTiles;

        _ = Parallel.For(
            0,
            tileCount,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            flat =>
            {
                long ti = flat / jTiles;
                long tj = flat - (ti * jTiles);

                long i0 = iFrom + (ti * iBlock);
                long j0 = jFrom + (tj * jBlock);

                body(i0, j0);
            });
    }
}