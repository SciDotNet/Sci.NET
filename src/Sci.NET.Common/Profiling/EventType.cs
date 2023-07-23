// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.Profiling;

/// <summary>
/// Event types for profiling.
/// </summary>
[PublicAPI]
public enum EventType
{
    /// <summary>
    /// A generic event.
    /// </summary>
    Generic = 0,

    /// <summary>
    /// A memory allocation event.
    /// </summary>
    MemoryAllocation = 1,

    /// <summary>
    /// A memory free event.
    /// </summary>
    MemoryFree = 2,

    /// <summary>
    /// A managed object dispose event.
    /// </summary>
    ObjectDisposed = 3,

    /// <summary>
    /// A kernel launch event.
    /// </summary>
    KernelLaunch = 4,

    /// <summary>
    /// A kernel finished event.
    /// </summary>
    KernelFinished = 5,
}