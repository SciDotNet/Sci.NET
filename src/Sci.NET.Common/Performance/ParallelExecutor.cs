// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.Performance;

/// <summary>
/// A helper class for executing loops in parallel.
/// </summary>
[PublicAPI]
public static class ParallelExecutor
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
}