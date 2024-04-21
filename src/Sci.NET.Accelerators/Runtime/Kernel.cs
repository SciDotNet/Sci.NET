// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common;

// ReSharper disable once CheckNamespace
namespace Sci.NET.Accelerators;

/// <summary>
/// Represents the parallel thread state.
/// </summary>
[PublicAPI]
public static class Kernel
{
    /// <summary>
    /// Gets the block dimension.
    /// </summary>
    public static Dim3 BlockDim { get; }

    /// <summary>
    /// Gets the grid dimension.
    /// </summary>
    public static Dim3 GridDim { get;  }

    /// <summary>
    /// Gets the block index.
    /// </summary>
    public static Dim3 BlockIdx { get; }

    /// <summary>
    /// Gets the thread index.
    /// </summary>
    public static Dim3 ThreadIdx { get; }

    /// <summary>
    /// Synchronizes the threads.
    /// </summary>
    public static void SyncThreads()
    {
        // Do nothing.
    }
}