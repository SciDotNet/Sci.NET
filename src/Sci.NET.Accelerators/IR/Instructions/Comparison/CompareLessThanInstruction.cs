﻿// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.Extensions;

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
    public ImmutableArray<IrValue> Operands => ImmutableArray.Create(Left, Right);

    /// <inheritdoc />
    public required MsilInstruction<IMsilOperand>? MsilInstruction { get; init; }

    /// <inheritdoc />
    public required BasicBlock Block { get; init; }

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
    public StringBuilder WriteToIrString(StringBuilder builder)
    {
        return builder.AppendWritable(Result).Append(" = ").Append("clt ").AppendWritable(Left).Append(", ").AppendWritable(Right);
    }
}