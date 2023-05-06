// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;

namespace Sci.NET.Common.Concurrency;

/// <summary>
/// A class for a 2D thread partitioner for nested for loops.
/// </summary>
[PublicAPI]
public class TwoDimensionalPartitioner : Partitioner<Tuple<long, long>>
{
    private readonly long _iFromInclusive;
    private readonly long _iToExclusive;
    private readonly long _jFromInclusive;
    private readonly long _jToExclusive;
    private readonly int _numProcessors;

    /// <summary>
    /// Initializes a new instance of the <see cref="TwoDimensionalPartitioner"/> class.
    /// </summary>
    /// <param name="iFromInclusive">The index to start from (inclusive) for the outer loop.</param>
    /// <param name="iToExclusive">The index to iterate to (exclusive) for outer loop.</param>
    /// <param name="jFromInclusive">The index to start from (inclusive) for the inner loop.</param>
    /// <param name="jToExclusive">The index to iterate to (exclusive) for inner loop.</param>
    public TwoDimensionalPartitioner(
        long iFromInclusive,
        long iToExclusive,
        long jFromInclusive,
        long jToExclusive)
    {
        _iFromInclusive = iFromInclusive;
        _iToExclusive = iToExclusive;
        _jFromInclusive = jFromInclusive;
        _jToExclusive = jToExclusive;
        _numProcessors = Environment.ProcessorCount;
    }

    /// <inheritdoc />
    public override bool SupportsDynamicPartitions => true;

    /// <inheritdoc />
    public override IList<IEnumerator<Tuple<long, long>>> GetPartitions(int partitionCount)
    {
        var partitions = new List<IEnumerator<Tuple<long, long>>>(partitionCount);

        var iChunkSize = (_iToExclusive - _iFromInclusive) / _numProcessors;
        var jChunkSize = (_jToExclusive - _jFromInclusive) / _numProcessors;

        for (var p = 0; p < partitionCount; p++)
        {
            var iStart = _iFromInclusive + (p * iChunkSize);
            var iEnd = p == partitionCount - 1 ? _iToExclusive : iStart + iChunkSize;
            var jStart = _jFromInclusive + (p * jChunkSize);
            var jEnd = p == partitionCount - 1 ? _jToExclusive : jStart + jChunkSize;

            partitions.Add(GetChunkEnumerator(iStart, iEnd, jStart, jEnd));
        }

        return partitions;
    }

    /// <inheritdoc />
    public override IEnumerable<Tuple<long, long>> GetDynamicPartitions()
    {
        foreach (var enumerator in GetPartitions(_numProcessors))
        {
            while (enumerator.MoveNext())
            {
                yield return enumerator.Current;
            }
        }
    }

    private static IEnumerator<Tuple<long, long>> GetChunkEnumerator(long iStart, long iEnd, long jStart, long jEnd)
    {
        for (var i = iStart; i < iEnd; i++)
        {
            for (var j = jStart; j < jEnd; j++)
            {
                yield return Tuple.Create(i, j);
            }
        }
    }
}