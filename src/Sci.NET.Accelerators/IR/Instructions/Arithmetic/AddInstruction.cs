// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.Extensions;

namespace Sci.NET.Accelerators.IR.Instructions.Arithmetic;

/// <summary>
/// Represents an add instruction.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members.")]
public class AddInstruction : IValueYieldingInstruction
{
    /// <inheritdoc />
    public string Name => "add";

    /// <inheritdoc />
    public required MsilInstruction<IMsilOperand>? MsilInstruction { get; init; }

    /// <summary>
    /// Gets the left value.
    /// </summary>
    public required IrValue Left { get; init; }

    /// <summary>
    /// Gets the right value.
    /// </summary>
    public required IrValue Right { get; init; }

    /// <inheritdoc />
    public required IrValue Result { get; init; }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder, int indentLevel)
    {
        return builder.Append('%').Append(Result.Identifier).Append(" = add ").AppendWritable(Left).Append(", ").AppendWritable(Right);
    }
}