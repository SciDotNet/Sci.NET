﻿// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;

namespace Sci.NET.Accelerators.IR.Instructions.Terminators;

/// <summary>
/// A branch instruction.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members.")]
public class BranchInstruction : IBranchInstruction
{
    /// <inheritdoc />
    public string Name => "br";

    /// <inheritdoc />
    public ImmutableArray<IrValue> Operands => ImmutableArray<IrValue>.Empty;

    /// <inheritdoc />
    public required MsilInstruction<IMsilOperand>? MsilInstruction { get; init; }

    /// <inheritdoc />
    public required BasicBlock Block { get; init; }

    /// <summary>
    /// Gets the target of the branch.
    /// </summary>
    public required BasicBlock Target { get; init; }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder)
    {
        return builder.Append("br ").Append(Target.Name);
    }

    /// <inheritdoc />
    public IEnumerable<BasicBlock> GetAllTargets()
    {
        yield return Target;
    }
}