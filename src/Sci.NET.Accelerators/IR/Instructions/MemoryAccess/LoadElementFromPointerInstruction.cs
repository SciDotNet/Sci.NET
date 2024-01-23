// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.Extensions;

namespace Sci.NET.Accelerators.IR.Instructions.MemoryAccess;

/// <inheritdoc />
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members.")]
public class LoadElementFromPointerInstruction : IAssignmentInstruction
{
    /// <inheritdoc />
    public string Name => "load";

    /// <inheritdoc />
    public required MsilInstruction<IMsilOperand>? MsilInstruction { get; init; }

    /// <inheritdoc />
    public required IrValue Result { get; init; }

    /// <summary>
    /// Gets the pointer to load from.
    /// </summary>
#pragma warning disable CA1720
    public required IrValue Pointer { get; init; }
#pragma warning restore CA1720

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder, int indentLevel)
    {
        return builder.Append('%').Append(Result.Identifier).Append(" = ").Append(Name).Append(' ').AppendWritable(Result.Type).Append(", ").AppendWritable(Pointer);
    }
}