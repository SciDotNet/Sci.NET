// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Text;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;

namespace Sci.NET.Accelerators.IR.Instructions;

/// <inheritdoc />
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public class CallInstruction : IValueYieldingInstruction
{
    /// <inheritdoc />
    public string Name => "call";

    /// <inheritdoc />
    public ImmutableArray<IrValue> Operands => Arguments;

    /// <inheritdoc />
    public required MsilInstruction<IMsilOperand>? MsilInstruction { get; init; }

    /// <inheritdoc />
    public required IrValue Result { get; init; }

    /// <summary>
    /// Gets the method that the instruction is associated with.
    /// </summary>
    public required ImmutableArray<IrValue> Arguments { get; init; }

    /// <summary>
    /// Gets the method that the instruction is associated with.
    /// </summary>
    public required MethodBase MethodBase { get; init; }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder)
    {
        if (!Result.Type.Equals(IrType.Void))
        {
            _ = Result.WriteToIrString(builder)
                .Append(" = ");
        }

        _ = builder.Append(Name)
            .Append(' ')
            .Append(MethodBase.Name)
            .Append('(');

        if (Arguments.Length > 0)
        {
            _ = Arguments[0].WriteToIrString(builder);
            for (var i = 1; i < Arguments.Length; i++)
            {
                _ = builder.Append(", ");
                _ = Arguments[i].WriteToIrString(builder);
            }
        }

        _ = builder.Append(')');

        return builder;
    }
}