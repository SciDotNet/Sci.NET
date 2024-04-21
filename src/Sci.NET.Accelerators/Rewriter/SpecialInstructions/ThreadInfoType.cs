// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.Rewriter.SpecialInstructions;

/// <summary>
/// Represents the type of thread information.
/// </summary>
public enum ThreadInfoType
{
    /// <summary>
    /// The thread index.
    /// </summary>
    ThreadIdx = 0,

    /// <summary>
    /// The block index.
    /// </summary>
    BlockIdx = 1,

    /// <summary>
    /// The grid dimension.
    /// </summary>
    GridDim = 2,

    /// <summary>
    /// The block dimension.
    /// </summary>
    BlockDim = 3
}