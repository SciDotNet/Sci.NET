// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;

namespace Sci.NET.Common.Profiling;

/// <summary>
/// A collection of profiling utilities.
/// </summary>
public static class Profiler
{
    /// <summary>
    /// Gets or sets a value indicating whether profiling is enabled.
    /// </summary>
#if DEBUG
    public static bool IsEnabled { get; set; } = true;
#else
    public static bool IsEnabled { get; set; } = true;
#endif

    /// <summary>
    /// Logs a memory allocation event.
    /// </summary>
    /// <param name="message">The message.</param>
    /// <param name="objectHashCode">The hash code of the allocated object.</param>
    public static void LogAllocation(string message, int objectHashCode)
    {
        if (IsEnabled)
        {
            Debug.WriteLine($"{DateTime.UtcNow.Ticks}\t{EventType.MemoryAllocation}\t{objectHashCode}\t{message}");
        }
    }

    /// <summary>
    /// Logs a memory free event.
    /// </summary>
    /// <param name="message">The message.</param>
    /// <param name="objectHashCode">The hash code of the freed object.</param>
    public static void LogMemoryFree(string message, int objectHashCode)
    {
        if (IsEnabled)
        {
            Debug.WriteLine($"{DateTime.UtcNow.Ticks}\t{EventType.MemoryFree}\t{objectHashCode}\t{message}");
        }
    }

    /// <summary>
    /// Logs a managed object dispose event.
    /// </summary>
    /// <param name="message">The message.</param>
    /// <param name="objectHashCode">The hash code of the disposed object.</param>
    public static void LogObjectDisposed(string message, int objectHashCode)
    {
        if (IsEnabled)
        {
            Debug.WriteLine($"{DateTime.UtcNow.Ticks}\t{EventType.ObjectDisposed}\t{objectHashCode}\t{message}");
        }
    }

    /// <summary>
    /// Logs a generic event.
    /// </summary>
    /// <param name="message">The message.</param>
    public static void LogGeneric(string message)
    {
        if (IsEnabled)
        {
            Debug.WriteLine($"{DateTime.UtcNow.Ticks}\t{EventType.Generic}\t\t{message}");
        }
    }
}