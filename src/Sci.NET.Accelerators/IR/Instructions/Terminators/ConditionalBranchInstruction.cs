// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
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
public class ConditionalBranchInstruction : IConditionalBranchInstruction
{
    /// <inheritdoc />
    public string Name => "br_if";

    /// <inheritdoc />
    public ImmutableArray<IrValue> Operands => ImmutableArray.Create(Condition);

    /// <inheritdoc />
    public required MsilInstruction<IMsilOperand>? MsilInstruction { get; init; }

    /// <summary>
    /// Gets condition to branch on.
    /// </summary>
    public required IrValue Condition { get; init; }

    /// <summary>
    /// Gets the target of the branch.
    /// </summary>
    public required BasicBlock Target { get; init; }

    /// <summary>
    /// Gets the target of the branch.
    /// </summary>
    public required BasicBlock FalseTarget { get; init; }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder)
    {
        return builder.Append("br_if ").AppendWritable(Condition).Append(' ').Append(Target.Name).Append(", ").Append(FalseTarget.Name);
    }
}