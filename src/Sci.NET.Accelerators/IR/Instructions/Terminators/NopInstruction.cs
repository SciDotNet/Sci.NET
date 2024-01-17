// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sci.NET.Accelerators.Disassembly.Cfg;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.IR.Instructions.Terminators;

/// <summary>
/// Represents a no-operation instruction.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public readonly struct NopInstruction : IInstruction, IValueEquatable<NopInstruction>
{
    /// <inheritdoc />
    public IMsilControlFlowGraphNode? MsilInstruction { get; init; }

    /// <inheritdoc />
    public required bool IsLeader { get; init; }

    /// <inheritdoc />
    public required bool IsTerminator { get; init; }

    /// <inheritdoc />
    public static bool operator ==(NopInstruction left, NopInstruction right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(NopInstruction left, NopInstruction right)
    {
        return !left.Equals(right);
    }

    /// <inheritdoc />
    public void AddToIrString(StringBuilder builder, int indentLevel)
    {
        _ = builder.Append(' ', indentLevel * IInstruction.IndentLevel)
            .Append("nop");
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(NopInstruction other)
    {
        return Equals(MsilInstruction, other.MsilInstruction) && IsLeader == other.IsLeader && IsTerminator == other.IsTerminator;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is NopInstruction other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(MsilInstruction, IsLeader, IsTerminator);
    }
}