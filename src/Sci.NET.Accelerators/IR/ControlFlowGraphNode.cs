// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Globalization;
using Sci.NET.Accelerators.Disassembly;

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// Represents a control flow graph node.
/// </summary>
[PublicAPI]
public class ControlFlowGraphNode : IControlFlowGraphNode
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ControlFlowGraphNode"/> class.
    /// </summary>
    /// <param name="instruction">The instruction.</param>
    public ControlFlowGraphNode(Instruction<IOperand> instruction)
    {
        Instruction = instruction;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ControlFlowGraphNode"/> class.
    /// </summary>
    /// <param name="instruction">The instruction.</param>
    /// <param name="nextInstructions">The next instructions.</param>
    public ControlFlowGraphNode(Instruction<IOperand> instruction, IEnumerable<Instruction<IOperand>> nextInstructions)
    {
        Instruction = instruction;
        NextInstructions = nextInstructions.ToList();
    }

    /// <inheritdoc />
    public Instruction<IOperand> Instruction { get; }

    /// <inheritdoc />
    public IList<Instruction<IOperand>> NextInstructions { get; } = new List<Instruction<IOperand>>();

    /// <inheritdoc />
    public bool IsLeader { get; set; }

    /// <inheritdoc />
    public bool IsTerminator { get; set; }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"{Instruction} -> {string.Join(", ", NextInstructions.Select(x => x.Offset.ToString("x4", CultureInfo.CurrentCulture)))}";
    }
}