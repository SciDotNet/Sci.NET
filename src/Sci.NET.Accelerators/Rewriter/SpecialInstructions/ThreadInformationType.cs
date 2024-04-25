// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.Rewriter.SpecialInstructions;

/// <summary>
/// Represents the type of thread information.
/// </summary>
[PublicAPI]
public enum ThreadInformationType
{
    /// <summary>
    /// No thread information.
    /// </summary>
    None = 0,

    /// <summary>
    /// The X component of the thread index.
    /// </summary>
    ThreadIdxX = (Dim3ThreadInformationType.ThreadIdx + Dim3Field.X) << Dim3ThreadInformationType.ThreadIdx,

    /// <summary>
    /// The Y component of the thread index.
    /// </summary>
    ThreadIdxY = (Dim3ThreadInformationType.ThreadIdx + Dim3Field.Y) << Dim3ThreadInformationType.ThreadIdx,

    /// <summary>
    /// The Z component of the thread index.
    /// </summary>
    ThreadIdxZ = (Dim3ThreadInformationType.ThreadIdx + Dim3Field.Z) << Dim3ThreadInformationType.ThreadIdx,

    /// <summary>
    /// The X component of the block index.
    /// </summary>
    BlockIdxX = (Dim3ThreadInformationType.BlockIdx + Dim3Field.X) << Dim3ThreadInformationType.BlockIdx,

    /// <summary>
    /// The Y component of the block index.
    /// </summary>
    BlockIdxY = (Dim3ThreadInformationType.BlockIdx + Dim3Field.Y) << Dim3ThreadInformationType.BlockIdx,

    /// <summary>
    /// The Z component of the block index.
    /// </summary>
    BlockIdxZ = (Dim3ThreadInformationType.BlockIdx + Dim3Field.Z) << Dim3ThreadInformationType.BlockIdx,

    /// <summary>
    /// The X component of the block dimension.
    /// </summary>
    BlockDimX = (Dim3ThreadInformationType.BlockDim + Dim3Field.X) << Dim3ThreadInformationType.BlockDim,

    /// <summary>
    /// The Y component of the block dimension.
    /// </summary>
    BlockDimY = (Dim3ThreadInformationType.BlockDim + Dim3Field.Y) << Dim3ThreadInformationType.BlockDim,

    /// <summary>
    /// The Z component of the block dimension.
    /// </summary>
    BlockDimZ = (Dim3ThreadInformationType.BlockDim + Dim3Field.Z) << Dim3ThreadInformationType.BlockDim,

    /// <summary>
    /// The X component of the grid dimension.
    /// </summary>
    GridDimX = (Dim3ThreadInformationType.GridDim + Dim3Field.X) << Dim3ThreadInformationType.GridDim,

    /// <summary>
    /// The Y component of the grid dimension.
    /// </summary>
    GridDimY = (Dim3ThreadInformationType.GridDim + Dim3Field.Y) << Dim3ThreadInformationType.GridDim,

    /// <summary>
    /// The Z component of the grid dimension.
    /// </summary>
    GridDimZ = (Dim3ThreadInformationType.GridDim + Dim3Field.Z) << Dim3ThreadInformationType.GridDim
}