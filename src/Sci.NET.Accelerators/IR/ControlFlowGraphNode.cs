// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Globalization;
using Sci.NET.Accelerators.IR.Rewriter;

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// Represents a control flow graph node.
/// </summary>
[PublicAPI]
public class ControlFlowGraphNode
{
    /// <summary>
    /// Gets the instruction of the control flow graph node.
    /// </summary>
    public SsaInstruction Instruction { get; init; }

    /// <summary>
    /// Gets the next instructions of the control flow graph node.
    /// </summary>
    public IList<SsaInstruction> NextInstructions { get; init; } = new List<SsaInstruction>();

    /// <summary>
    /// Gets or sets a value indicating whether this node is a leader.
    /// </summary>
    public bool IsLeader { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether this node is a terminator.
    /// </summary>
    public bool IsTerminator { get; set; }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"{Instruction} -> {string.Join(", ", NextInstructions.Select(x => x.Offset.ToString("x4", CultureInfo.CurrentCulture)))}";
    }
}