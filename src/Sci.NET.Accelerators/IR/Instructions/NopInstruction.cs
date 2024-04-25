// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Text;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.IR.Instructions;

/// <summary>
/// A no-operation instruction.
/// </summary>
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required properties.")]
public readonly struct NopInstruction : IInstruction, IValueEquatable<NopInstruction>
{
    /// <inheritdoc />
    public string Name => "nop";

    /// <inheritdoc />
    public ImmutableArray<IrValue> Operands => ImmutableArray<IrValue>.Empty;

    /// <inheritdoc />
    public required MsilInstruction<IMsilOperand>? MsilInstruction { get; init; }

    /// <inheritdoc />
    public required BasicBlock Block { get; init; }

    /// <inheritdoc />
    public static bool operator ==(NopInstruction left, NopInstruction right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(NopInstruction left, NopInstruction right)
    {
        return !(left == right);
    }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder)
    {
        return builder;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(NopInstruction other)
    {
        return true;
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is NopInstruction other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return 1;
    }
}