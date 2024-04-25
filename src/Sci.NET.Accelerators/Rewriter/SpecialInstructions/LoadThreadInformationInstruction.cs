// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.IR;
using Sci.NET.Accelerators.IR.Instructions;

namespace Sci.NET.Accelerators.Rewriter.SpecialInstructions;

/// <summary>
/// Represents an instruction that loads thread information.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public class LoadThreadInformationInstruction : IValueYieldingInstruction
{
    /// <inheritdoc />
    public string Name => $"load_thread_info_{Type}";

    /// <inheritdoc />
    public ImmutableArray<IrValue> Operands => ImmutableArray<IrValue>.Empty;

    /// <inheritdoc />
    public required MsilInstruction<IMsilOperand>? MsilInstruction { get; init; }

    /// <inheritdoc />
    public required IrValue Result { get; init; }

    /// <summary>
    /// Gets the type of the thread information.
    /// </summary>
    public required ThreadInformationType Type { get; init; }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder)
    {
        _ = Result.WriteToIrString(builder)
            .Append(" = ")
            .Append(Name);

        return builder;
    }
}