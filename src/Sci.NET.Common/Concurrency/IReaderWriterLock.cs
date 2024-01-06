// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.Concurrency;

/// <summary>
/// An interface for a reader-writer lock.
/// </summary>
[PublicAPI]
public interface IReaderWriterLock
{
    /// <summary>
    /// Gets the average time a reader lock waited in the queue.
    /// </summary>
    public double AverageReaderLockQueueTime { get; }

    /// <summary>
    /// Gets the average time a writer lock waited in the queue.
    /// </summary>
    public double AverageWriterLockQueueTime { get; }

    /// <summary>
    /// Gets the maximum time a reader lock waited in the queue.
    /// </summary>
    public double MaxReaderLockQueueTime { get; }

    /// <summary>
    /// Gets the maximum time a writer lock waited in the queue.
    /// </summary>
    public double MaxWriterLockQueueTime { get; }

    /// <summary>
    /// Gets the minimum time a reader lock waited in the queue.
    /// </summary>
    public double TotalReaderLockQueueTime { get; }

    /// <summary>
    /// Gets the minimum time a writer lock waited in the queue.
    /// </summary>
    public double TotalWriterLockQueueTime { get; }

    /// <summary>
    /// Gets the total time a lock waited in the queue.
    /// </summary>
    public double TotalQueueTime { get; }

    /// <summary>
    /// Gets the number of reader locks.
    /// </summary>
    public int ReaderLockCount { get; }

    /// <summary>
    /// Gets the number of writer locks.
    /// </summary>
    public int WriterLockCount { get; }

    /// <summary>
    /// Acquires a reader lock.
    /// </summary>
    /// <param name="timeout">The timeout for acquiring the lock.</param>
    public void AcquireReaderLock(TimeSpan timeout);

    /// <summary>
    /// Acquires a writer lock.
    /// </summary>
    /// <param name="timeout">The timeout for acquiring the lock.</param>
    public void AcquireWriterLock(TimeSpan timeout);

    /// <summary>
    /// Releases a reader lock.
    /// </summary>
    public void ReleaseReaderLock();

    /// <summary>
    /// Releases a writer lock.
    /// </summary>
    public void ReleaseWriterLock();
}