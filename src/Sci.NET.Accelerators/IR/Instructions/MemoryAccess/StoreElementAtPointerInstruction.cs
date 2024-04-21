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
/// Represents a store element at pointer instruction.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members.")]
public class StoreElementAtPointerInstruction : IInstruction
{
    /// <inheritdoc />
    public string Name => "store";

    /// <inheritdoc />
    public ImmutableArray<IrValue> Operands => ImmutableArray.Create(Value, Pointer);

    /// <inheritdoc />
    public required MsilInstruction<IMsilOperand>? MsilInstruction { get; init; }

    /// <summary>
    /// Gets the value to store.
    /// </summary>
    public required IrValue Value { get; init; }

    /// <summary>
    /// Gets the pointer to store to.
    /// </summary>
#pragma warning disable CA1720
    public required IrValue Pointer { get; init; }
#pragma warning restore CA1720

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder)
    {
        return builder.Append("store ").AppendWritable(Value.Type).Append(' ').AppendWritable(Value).Append(", ").AppendWritable(Pointer);
    }
}