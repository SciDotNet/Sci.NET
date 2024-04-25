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

namespace Sci.NET.Accelerators.Rewriter.SpecialInstructions;

/// <summary>
/// Represents a special instruction that loads the dimensions of a 3D thread info.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public class LoadDim3ThreadInfoInstruction : IValueYieldingInstruction
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LoadDim3ThreadInfoInstruction"/> class.
    /// </summary>
    /// <param name="callInstruction">The method operand.</param>
    /// <param name="dim3ThreadIdx">The thread info type.</param>
    [SetsRequiredMembers]
    public LoadDim3ThreadInfoInstruction(CallInstruction callInstruction, Dim3ThreadInformationType dim3ThreadIdx)
    {
        Method = callInstruction.MethodBase;
        Type = dim3ThreadIdx;
        Arguments = callInstruction.Arguments;
        Result = callInstruction.Result;
        MsilInstruction = callInstruction.MsilInstruction;
        Block = callInstruction.Block;
    }

    /// <inheritdoc />
    public string Name => "load_dim3_thread_info";

    /// <summary>
    /// Gets the method that the instruction is associated with.
    /// </summary>
    public required MethodBase Method { get; init; }

    /// <summary>
    /// Gets the type of the thread info.
    /// </summary>
    public required Dim3ThreadInformationType Type { get; init; }

    /// <inheritdoc />
    public ImmutableArray<IrValue> Operands => ImmutableArray.Create(Result).AddRange(Arguments);

    /// <summary>
    /// Gets the method that the instruction is associated with.
    /// </summary>
    public required ImmutableArray<IrValue> Arguments { get; init; }

    /// <inheritdoc />
    public required MsilInstruction<IMsilOperand>? MsilInstruction { get; init; }

    /// <inheritdoc />
    public required BasicBlock Block { get; init; }

    /// <inheritdoc />
    public required IrValue Result { get; init; }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder)
    {
        _ = builder
            .AppendWritable(Result)
            .Append(" = ")
            .Append(Name)
            .Append('(')
            .AppendJoin(", ", Arguments.Select(x => x.ToString()))
            .Append(')');

        return builder;
    }
}