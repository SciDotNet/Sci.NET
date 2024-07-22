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
public class BranchGreaterThanInstruction : IConditionalBranchInstruction
{
    /// <inheritdoc />
    public string Name => "bgt";

    /// <inheritdoc />
    public ImmutableArray<IrValue> Operands => ImmutableArray.Create(Left, Right);

    /// <inheritdoc />
    public required MsilInstruction<IMsilOperand>? MsilInstruction { get; init; }

    /// <inheritdoc />
    public required BasicBlock Block { get; init; }

    /// <inheritdoc />
    public required BasicBlock Target { get; init; }

    /// <inheritdoc/>
    public required BasicBlock FalseTarget { get; init; }

    /// <summary>
    /// Gets the left value to compare.
    /// </summary>
    public required IrValue Left { get; init; }

    /// <summary>
    /// Gets the right value to compare.
    /// </summary>
    public required IrValue Right { get; init; }

    /// <inheritdoc />
    public IEnumerable<BasicBlock> GetAllTargets()
    {
        yield return Target;
        yield return FalseTarget;
    }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder)
    {
        return builder.Append("bgt ").AppendWritable(Left).Append(" < ").AppendWritable(Right).Append(' ').Append(Target.Name).Append(", ")
            .Append(FalseTarget.Name);
    }
}