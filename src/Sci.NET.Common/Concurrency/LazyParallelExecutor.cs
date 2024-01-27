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
    public static long For(long fromInclusive, long toExclusive, long parallelizationThreshold, long increment, Action<long> body)
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
        else
        {
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
    public static (long iCount, long jCount) For(
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
        var i = iFromInclusive;
        var j = jFromInclusive;

        if (loopIterations < parallelizationThreshold)
        {
            for (; i < iToExclusive; i += iIncrement)
            {
                j = 0;

                for (; j < jToExclusive; j += jIncrement)
                {
                    body(i, j);
                }
            }
        }
        else
        {
            var partitioner = new TwoDimensionalPartitioner(
                iFromInclusive,
                iToExclusive,
                jFromInclusive,
                jToExclusive,
                iIncrement,
                jIncrement);
            _ = Parallel.ForEach(partitioner, pair => body(pair.Item1, pair.Item2));
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
    public static (long iCount, long jCount) For(
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
}