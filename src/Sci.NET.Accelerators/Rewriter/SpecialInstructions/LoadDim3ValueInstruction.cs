// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Text;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.Extensions;
using Sci.NET.Accelerators.IR;
using Sci.NET.Accelerators.IR.Instructions;
using Sci.NET.Common;

namespace Sci.NET.Accelerators.Rewriter.SpecialInstructions;

/// <summary>
/// Represents an instruction that loads a <see cref="Dim3"/> value.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public class LoadDim3ValueInstruction : IValueYieldingInstruction
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LoadDim3ValueInstruction"/> class.
    /// </summary>
    /// <param name="callInstruction">The method operand.</param>
    /// <param name="dim3Field">The field of the <see cref="Dim3"/> value.</param>
    [SetsRequiredMembers]
    public LoadDim3ValueInstruction(CallInstruction callInstruction, Dim3Field dim3Field)
    {
        Method = callInstruction.MethodBase;
        Field = dim3Field;
        Arguments = callInstruction.Arguments;
        Result = callInstruction.Result;
        MsilInstruction = callInstruction.MsilInstruction;
    }

    /// <inheritdoc />
    public string Name => "load_dim3_value";

    /// <summary>
    /// Gets the method that the instruction is associated with.
    /// </summary>
    public required MethodBase Method { get; init; }

    /// <summary>
    /// Gets the field of the <see cref="Dim3"/> value.
    /// </summary>
    public required Dim3Field Field { get; init; }

    /// <inheritdoc />
    public ImmutableArray<IrValue> Operands => ImmutableArray.Create(Result).AddRange(Arguments);

    /// <summary>
    /// Gets the arguments of the instruction.
    /// </summary>
    public required ImmutableArray<IrValue> Arguments { get; init; }

    /// <inheritdoc />
    public required MsilInstruction<IMsilOperand>? MsilInstruction { get; init; }

    /// <inheritdoc />
    public required IrValue Result { get; init; }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder)
    {
        var args = Arguments.Select(x =>
        {
            var sb = new StringBuilder();
            _ = x.WriteToIrString(sb);
            return sb.ToString();
        });

        _ = builder
            .AppendWritable(Result)
            .Append(" = ")
            .Append(Name)
            .Append('(')
            .AppendJoin(", ", args)
            .Append(')');

        return builder;
    }
}