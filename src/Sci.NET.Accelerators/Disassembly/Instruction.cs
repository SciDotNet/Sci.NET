// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection.Emit;

namespace Sci.NET.Accelerators.Disassembly;

/// <summary>
/// Represents an MSIL instruction.
/// </summary>
[PublicAPI]
public class Instruction
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Instruction"/> class.
    /// </summary>
    /// <param name="offset">The offset of the instruction.</param>
    /// <param name="opCode">The opcode of the instruction.</param>
    public Instruction(int offset, OpCode opCode)
    {
        Offset = offset;
        OpCode = opCode;
    }

    /// <summary>
    /// Gets or sets the offset of the instruction.
    /// </summary>
    public int Offset { get; set; }

    /// <summary>
    /// Gets or sets the opcode of the instruction.
    /// </summary>
    public OpCode OpCode { get; set; }

    /// <summary>
    /// Gets or sets the operand of the instruction.
    /// </summary>
    public object? Operand { get; set; }
}