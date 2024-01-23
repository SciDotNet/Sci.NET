// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;

namespace Sci.NET.Accelerators.IR.Instructions.Comparison;

/// <summary>
/// An instruction that compares two values for less than.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members.")]
public class CompareLessThanInstruction : IValueYieldingInstruction
{
    /// <inheritdoc />
    public string Name => "clt";

    /// <inheritdoc />
    public required MsilInstruction<IMsilOperand>? MsilInstruction { get; init; }

    /// <inheritdoc />
    public required IrValue Result { get; init; }

    /// <summary>
    /// Gets the left operand.
    /// </summary>
    public required IrValue Left { get; init; }

    /// <summary>
    /// Gets the right operand.
    /// </summary>
    public required IrValue Right { get; init; }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder, int indentLevel)
    {
        return builder.Append('%').Append(Result.Identifier).Append(" = ").Append("clt ").Append('%').Append(Left.Identifier).Append(", ").Append('%').Append(Right.Identifier);
    }
}