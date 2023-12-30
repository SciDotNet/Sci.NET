// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.IR.Rewriter.Variables;

/// <summary>
/// Represents an operand type.
/// </summary>
[PublicAPI]
public enum SsaOperandType
{
    /// <summary>
    /// A void operand.
    /// </summary>
    None = -1,

    /// <summary>
    /// An argument.
    /// </summary>
    Argument = 0,

    /// <summary>
    /// A local variable.
    /// </summary>
    Local = 1,

    /// <summary>
    /// A temporary value.
    /// </summary>
    Temp = 2,

    /// <summary>
    /// A constant value.
    /// </summary>
    Constant = 3,

    /// <summary>
    /// The program counter.
    /// </summary>
    ProgramCounter = 4,
}