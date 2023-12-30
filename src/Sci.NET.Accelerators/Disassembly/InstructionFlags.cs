// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace Sci.NET.Accelerators.Disassembly;

/// <summary>
/// Instruction flags.
/// </summary>
[Flags]
[SuppressMessage("Naming", "CA1720:Identifier contains type name", Justification = "DotNET naming convention.")]
[SuppressMessage("Naming", "CA1711:Identifiers should not have incorrect suffix", Justification = "DotNET naming convention.")]
public enum InstructionFlags
{
    /// <summary>
    /// None.
    /// </summary>
    None = 0,

    /// <summary>
    /// Unsigned operation.
    /// </summary>
    Unsigned = 1 << 0,

    /// <summary>
    /// Overflow check requested.
    /// </summary>
    Overflow = 1 << 1,

    /// <summary>
    /// Unchecked operation.
    /// </summary>
    Unchecked = 1 << 2,

    /// <summary>
    /// Unaligned operation.
    /// </summary>
    Unaligned = 1 << 3,

    /// <summary>
    /// Volatile access.
    /// </summary>
    Volatile = 1 << 4,

    /// <summary>
    /// ReadOnly access.
    /// </summary>
    ReadOnly = 1 << 5,

    /// <summary>
    /// Tail call.
    /// </summary>
    Tail = 1 << 6,

    /// <summary>
    /// Constraint virtual-function access.
    /// </summary>
    Constrained = 1 << 7,
}