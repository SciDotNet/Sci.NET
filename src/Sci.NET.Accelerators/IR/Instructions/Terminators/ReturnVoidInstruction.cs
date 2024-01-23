// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;

namespace Sci.NET.Accelerators.IR.Instructions.Terminators;

/// <inheritdoc />
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members.")]
public class ReturnVoidInstruction : IInstruction
{
    /// <inheritdoc />
    public string Name => "ret";

    /// <inheritdoc />
    public required MsilInstruction<IMsilOperand>? MsilInstruction { get; init; }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder, int indentLevel)
    {
        return builder.Append("ret");
    }
}