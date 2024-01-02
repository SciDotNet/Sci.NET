// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Disassembly.Operands;

namespace Sci.NET.Accelerators.Disassembly.Cfg;

/// <summary>
/// Represents a control flow graph node.
/// </summary>
[PublicAPI]
public interface IMsilControlFlowGraphNode
{
    /// <summary>
    /// Gets the instruction.
    /// </summary>
    public Instruction<IOperand> Instruction { get; }

    /// <summary>
    /// Gets the next instructions.
    /// </summary>
    public IList<Instruction<IOperand>> NextInstructions { get; }

    /// <summary>
    /// Gets or sets a value indicating whether this node is a leader.
    /// </summary>
    public bool IsLeader { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether this node is a terminator.
    /// </summary>
    public bool IsTerminator { get; set; }
}