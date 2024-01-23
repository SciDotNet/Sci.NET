// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.Extensions;

namespace Sci.NET.Accelerators.IR.Instructions.Terminators;

/// <summary>
/// A conditional branch instruction.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members.")]
public class ConditionalBranchInstruction : IInstruction
{
    /// <inheritdoc />
    public string Name => "br_if";

    /// <inheritdoc />
    public required MsilInstruction<IMsilOperand>? MsilInstruction { get; init; }

    /// <summary>
    /// Gets condition to branch on.
    /// </summary>
    public required IrValue Condition { get; init; }

    /// <summary>
    /// Gets the target of the branch.
    /// </summary>
    public required BasicBlock TargetTrue { get; init; }

    /// <summary>
    /// Gets the target of the branch.
    /// </summary>
    public required BasicBlock TargetFalse { get; init; }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder, int indentLevel)
    {
        return builder.Append("br_if ").AppendWritable(Condition).Append(' ').Append(TargetTrue.Name).Append(", ").Append(TargetFalse.Name);
    }
}