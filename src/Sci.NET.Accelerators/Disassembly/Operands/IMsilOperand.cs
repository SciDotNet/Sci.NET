// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection.Emit;

namespace Sci.NET.Accelerators.Disassembly.Operands;

/// <summary>
/// An interface for a MSIL instruction operand.
/// </summary>
[PublicAPI]
public interface IMsilOperand
{
    /// <summary>
    /// Gets the operand type.
    /// </summary>
    public OperandType OperandType { get; }
}