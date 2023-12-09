// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection.Emit;

namespace Sci.NET.Accelerators.Disassembly.Instructions;

/// <summary>
/// Represents an instruction with an operand.
/// </summary>
/// <typeparam name="TOperand">The operand type.</typeparam>
[PublicAPI]
public interface IInstruction<out TOperand>
    where TOperand : IOperand
{
    /// <summary>
    /// Gets the OpCode of the instruction.
    /// </summary>
    public OpCode OpCode { get; }

    /// <summary>
    /// Gets the offset of the instruction.
    /// </summary>
    public int Offset { get; }

    /// <summary>
    /// Gets the size of the instruction.
    /// </summary>
    public int Size { get; }

    /// <summary>
    /// Gets the operand.
    /// </summary>
    public TOperand Operand { get; }

    /// <summary>
    /// Gets a value indicating whether the instruction is a terminator.
    /// </summary>
    public bool IsTerminator { get; }

    /// <summary>
    /// Gets a value indicating whether the instruction is a branch.
    /// </summary>
    public bool IsBranch { get; }

    /// <summary>
    /// Gets a value indicating whether the instruction is an unconditional branch.
    /// </summary>
    public bool IsUnconditionalBranch { get; }

    /// <summary>
    /// Gets a value indicating whether the instruction is a conditional branch.
    /// </summary>
    public bool IsConditionalBranch { get; }
}