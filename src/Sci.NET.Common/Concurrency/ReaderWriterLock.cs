// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;

namespace Sci.NET.Common.Concurrency;

/// <summary>
/// A reader-writer lock.
/// </summary>
[PublicAPI]
public class ReaderWriterLock : IReaderWriterLock
{
    private readonly System.Threading.ReaderWriterLock _lock;
    private readonly List<long> _readerLockQueueTimes;
    private readonly List<long> _writerLockQueueTimes;
    private volatile int _readerLockCount;
    private volatile int _writerLockCount;

    /// <summary>
    /// Initializes a new instance of the <see cref="ReaderWriterLock"/> class.
    /// </summary>
    public ReaderWriterLock()
    {
        _lock = new System.Threading.ReaderWriterLock();
        _readerLockQueueTimes = new List<long>();
        _writerLockQueueTimes = new List<long>();
    }

    /// <inheritdoc />
    public double AverageReaderLockQueueTime => _readerLockQueueTimes.Average();

    /// <inheritdoc />
    public double AverageWriterLockQueueTime => _writerLockQueueTimes.Average();

    /// <inheritdoc />
    public double MaxReaderLockQueueTime => _readerLockQueueTimes.Max();

    /// <inheritdoc />
    public double MaxWriterLockQueueTime => _writerLockQueueTimes.Max();

    /// <inheritdoc />
    public double TotalReaderLockQueueTime => _readerLockQueueTimes.Sum();

    /// <inheritdoc />
    public double TotalWriterLockQueueTime => _writerLockQueueTimes.Sum();

    /// <inheritdoc />
    public double TotalQueueTime => TotalReaderLockQueueTime + TotalWriterLockQueueTime;

    /// <inheritdoc />
    public int ReaderLockCount => _readerLockCount;

    /// <inheritdoc />
    public int WriterLockCount => _writerLockCount;

    /// <inheritdoc />
    public void AcquireReaderLock(TimeSpan timeout)
    {
        var sw = Stopwatch.StartNew();
        _lock.AcquireReaderLock(timeout);
        sw.Stop();
        _readerLockQueueTimes.Add(sw.Elapsed.Ticks);
        _ = Interlocked.Increment(ref _readerLockCount);
    }

    /// <inheritdoc />
    public void AcquireWriterLock(TimeSpan timeout)
    {
        var sw = Stopwatch.StartNew();
        _lock.AcquireWriterLock(timeout);
        sw.Stop();
        _writerLockQueueTimes.Add(sw.Elapsed.Ticks);
        _ = Interlocked.Increment(ref _writerLockCount);
    }

    /// <inheritdoc />
    public void ReleaseReaderLock()
    {
        _lock.ReleaseReaderLock();
        _ = Interlocked.Decrement(ref _readerLockCount);
    }

    /// <inheritdoc />
    public void ReleaseWriterLock()
    {
        _lock.ReleaseWriterLock();
        _ = Interlocked.Decrement(ref _writerLockCount);
    }
}