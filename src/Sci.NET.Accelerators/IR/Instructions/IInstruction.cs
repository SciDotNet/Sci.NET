// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Text;
using Sci.NET.Accelerators.Disassembly.Cfg;

namespace Sci.NET.Accelerators.IR.Instructions;

/// <summary>
/// Represents an instruction.
/// </summary>
[PublicAPI]
public interface IInstruction
{
    /// <summary>
    /// Gets the indent count.
    /// </summary>
    public const int IndentLevel = 2;

    /// <summary>
    /// Gets the MSIL instruction which this instruction represents.
    /// </summary>
    public IMsilControlFlowGraphNode? MsilInstruction { get; }

    /// <summary>
    /// Gets a value indicating whether this instruction has an MSIL instruction equivalent.
    /// </summary>
    public bool HasMsilInstructionEquivalent => MsilInstruction is not null;

    /// <summary>
    /// Gets a value indicating whether the symbol is a leader.
    /// </summary>
    public bool IsLeader { get; init; }

    /// <summary>
    /// Gets a value indicating whether the symbol is a terminator.
    /// </summary>
    public bool IsTerminator { get; init; }

    /// <summary>
    /// Gets the LLVM type of the instruction.
    /// </summary>
    /// <param name="builder">The string builder to append to.</param>
    /// <param name="indentLevel">The indentation level.</param>
    public void AddToIrString(StringBuilder builder, int indentLevel);
}