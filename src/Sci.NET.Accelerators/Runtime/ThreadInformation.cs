// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

// ReSharper disable once CheckNamespace
namespace Sci.NET.Accelerators;

/// <summary>
/// Represents thread information.
/// </summary>
[PublicAPI]
public enum ThreadInformation
{
    /// <summary>
    /// No thread information.
    /// </summary>
    None = 0,

    /// <summary>
    /// The X thread index.
    /// </summary>
    ThreadIdxX = 1,

    /// <summary>
    /// The Y thread index.
    /// </summary>
    ThreadIdxY = 2,

    /// <summary>
    /// The Z thread index.
    /// </summary>
    ThreadIdxZ = 3,

    /// <summary>
    /// The X block index.
    /// </summary>
    BlockIdxX = 4,

    /// <summary>
    /// The Y block index.
    /// </summary>
    BlockIdxY = 5,

    /// <summary>
    /// The Z block index.
    /// </summary>
    BlockIdxZ = 6,

    /// <summary>
    /// The X block dimension.
    /// </summary>
    BlockDimX = 7,

    /// <summary>
    /// The Y block dimension.
    /// </summary>
    BlockDimY = 8,

    /// <summary>
    /// The Z block dimension.
    /// </summary>
    BlockDimZ = 9,

    /// <summary>
    /// The X grid dimension.
    /// </summary>
    GridDimX = 10,

    /// <summary>
    /// The Y grid dimension.
    /// </summary>
    GridDimY = 11,

    /// <summary>
    /// The Z grid dimension.
    /// </summary>
    GridDimZ = 12
}