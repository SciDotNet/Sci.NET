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
/// An instruction that loads an argument.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members.")]
public class LoadArgumentInstruction : IAssignmentInstruction
{
    /// <inheritdoc />
    public string Name => "load_argument";

    /// <inheritdoc />
    public ImmutableArray<IrValue> Operands => ImmutableArray.Create(Parameter);

    /// <inheritdoc />
    public required MsilInstruction<IMsilOperand>? MsilInstruction { get; init; }

    /// <inheritdoc />
    public required BasicBlock Block { get; init; }

    /// <summary>
    /// Gets the parameter to load.
    /// </summary>
    public required IrValue Parameter { get; init; }

    /// <inheritdoc />
    public required IrValue Result { get; init; }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder)
    {
        return builder.AppendWritable(Result).Append(" = ").AppendWritable(Parameter);
    }
}