// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sci.NET.Accelerators.Disassembly.Cfg;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.IR.Instructions;

/// <summary>
/// Represents an alloca instruction.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public readonly struct AllocaInstruction : IInstructionWithResult, IValueEquatable<AllocaInstruction>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="AllocaInstruction"/> struct.
    /// </summary>
    /// <param name="variable">The type of the variable.</param>
    /// <param name="isLeader">A value indicating whether the symbol is a leader.</param>
    /// <param name="isTerminator">A value indicating whether the symbol is a terminator.</param>
    /// <param name="msilInstruction">The MSIL instruction which this instruction represents.</param>
    public AllocaInstruction(Variable variable, bool isLeader, bool isTerminator, IMsilControlFlowGraphNode? msilInstruction = null)
    {
        Result = variable;
        Result = variable;
        MsilInstruction = msilInstruction;
        IsLeader = isLeader;
        IsTerminator = isTerminator;
    }

    /// <inheritdoc />
    public IMsilControlFlowGraphNode? MsilInstruction { get; }

    /// <inheritdoc />
    public required bool IsLeader { get; init; }

    /// <inheritdoc />
    public required bool IsTerminator { get; init; }

    /// <inheritdoc />
    public required Variable Result { get; init; }

    /// <summary>
    /// Gets the name of the variable.
    /// </summary>
    public string Name => Result.Name;

    /// <summary>
    /// Gets the type of the variable.
    /// </summary>
    public Type Type => Result.Type;

    /// <summary>
    /// Gets the LLVM type of the variable.
    /// </summary>
    public LlvmCompatibleTypes LlvmType => Type.ToLlvmType();

    /// <inheritdoc />
    public static bool operator ==(AllocaInstruction left, AllocaInstruction right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(AllocaInstruction left, AllocaInstruction right)
    {
        return !(left == right);
    }

    /// <inheritdoc />
    public void AddToIrString(StringBuilder builder, int indentLevel)
    {
        _ = builder
            .Append(' ', indentLevel * IInstruction.IndentLevel)
            .Append(Result.LlvmType.GetCompilerString())
            .Append(' ')
            .Append(Result.Name)
            .Append(" = alloca ")
            .AppendLine(Result.LlvmType.GetCompilerString());
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(AllocaInstruction other)
    {
        return Equals(MsilInstruction, other.MsilInstruction) && Result.Equals(other.Result) && Name == other.Name && Type == other.Type;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is AllocaInstruction other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(
            MsilInstruction,
            Result,
            Name,
            Type);
    }
}