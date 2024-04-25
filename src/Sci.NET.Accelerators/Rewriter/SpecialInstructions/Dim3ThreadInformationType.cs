// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.Rewriter.SpecialInstructions;

/// <summary>
/// Represents the type of thread information.
/// </summary>
public enum Dim3ThreadInformationType
{
    /// <summary>
    /// No thread information.
    /// </summary>
    None = 0,

    /// <summary>
    /// The thread index.
    /// </summary>
    ThreadIdx = 1,

    /// <summary>
    /// The block index.
    /// </summary>
    BlockIdx = 2,

    /// <summary>
    /// The grid dimension.
    /// </summary>
    GridDim = 3,

    /// <summary>
    /// The block dimension.
    /// </summary>
    BlockDim = 4
}