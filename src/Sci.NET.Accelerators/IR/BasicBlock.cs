// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Disassembly.Instructions;

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// Represents a basic block.
/// </summary>
[PublicAPI]
public class BasicBlock
{
    /// <summary>
    /// Gets the instructions of the basic block.
    /// </summary>
    public IList<Instruction<IOperand>> Instructions { get; init; } = new List<Instruction<IOperand>>();

    /// <summary>
    /// Gets the next basic blocks which can be reached from this basic block.
    /// </summary>
    public IList<BasicBlock> NextBlocks { get; init; } = new List<BasicBlock>();

    /// <summary>
    /// Gets the start offset of the basic block.
    /// </summary>
    public int StartOffset { get; init; }

    /// <summary>
    /// Gets or sets the end offset of the basic block.
    /// </summary>
    public int EndOffset { get; set; }
}