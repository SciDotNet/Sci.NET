﻿// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace Sci.NET.Accelerators.Disassembly.Cfg;

/// <summary>
/// Represents a control flow graph node.
/// </summary>
[PublicAPI]
public class MsilControlFlowGraphNode : IMsilControlFlowGraphNode
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MsilControlFlowGraphNode"/> class.
    /// </summary>
    /// <param name="instruction">The instruction.</param>
    public MsilControlFlowGraphNode(Instruction<IOperand> instruction)
    {
        Instruction = instruction;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="MsilControlFlowGraphNode"/> class.
    /// </summary>
    /// <param name="instruction">The instruction.</param>
    /// <param name="nextInstructions">The next instructions.</param>
    public MsilControlFlowGraphNode(Instruction<IOperand> instruction, IEnumerable<Instruction<IOperand>> nextInstructions)
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