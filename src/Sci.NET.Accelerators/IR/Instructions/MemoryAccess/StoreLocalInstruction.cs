// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.Extensions;

namespace Sci.NET.Accelerators.IR.Instructions.MemoryAccess;

/// <summary>
/// An instruction that stores a value to a local variable.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members.")]
public class StoreLocalInstruction : IInstruction
{
    /// <summary>
    /// Gets the name of the instruction.
    /// </summary>
    public string Name => "store_local";

    /// <inheritdoc />
    public ImmutableArray<IrValue> Operands => ImmutableArray.Create(Value, Local);

    /// <inheritdoc />
    public required MsilInstruction<IMsilOperand>? MsilInstruction { get; init; }

    /// <inheritdoc />
    public required BasicBlock Block { get; init; }

    /// <summary>
    /// Gets the local variable to store to.
    /// </summary>
    public required IrValue Value { get; init; }

    /// <summary>
    /// Gets the value to store.
    /// </summary>
    public required IrValue Local { get; init; }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder)
    {
        return builder.Append("store ").AppendWritable(Local).Append(", ").Append(Value.Identifier);
    }
}