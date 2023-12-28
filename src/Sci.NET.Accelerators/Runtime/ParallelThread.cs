// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common;

namespace Sci.NET.Accelerators.Runtime;

/// <summary>
/// Represents a thread in the parallel runtime.
/// </summary>
[PublicAPI]
public static class ParallelThread
{
    /// <summary>
    /// Gets the thread index.
    /// </summary>
    public static Dim3 ThreadIdx { get; }

    /// <summary>
    /// Gets the block index.
    /// </summary>
    public static Dim3 BlockDim { get; }
}