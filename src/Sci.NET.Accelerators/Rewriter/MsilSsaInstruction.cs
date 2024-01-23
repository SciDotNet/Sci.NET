// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.Rewriter.Variables;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Rewriter;

/// <summary>
/// Represents a symbol.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public readonly struct MsilSsaInstruction : IValueEquatable<MsilSsaInstruction>
{
    /// <summary>
    /// Gets the operands of the symbol.
    /// </summary>
    public required IReadOnlyList<ISsaVariable> Operands { get; init; }

    /// <summary>
    /// Gets the instruction of the symbol.
    /// </summary>
    public required MsilInstruction<IMsilOperand> MsilInstruction { get; init; }

    /// <summary>
    /// Gets the instruction of the symbol.
    /// </summary>
    public required OpCodeTypes IlOpCode { get; init; }

    /// <summary>
    /// Gets the result of the symbol.
    /// </summary>
    public required ISsaVariable Result { get; init; }

    /// <summary>
    /// Gets the offset of the symbol.
    /// </summary>
    public required int Offset { get; init; }

    /// <inheritdoc />
    public static bool operator ==(MsilSsaInstruction left, MsilSsaInstruction right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(MsilSsaInstruction left, MsilSsaInstruction right)
    {
        return !(left == right);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is MsilSsaInstruction other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(MsilSsaInstruction other)
    {
        return Operands.Equals(other.Operands) && IlOpCode.Equals(other.IlOpCode);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(Operands, IlOpCode);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        if (Result is null or VoidSsaVariable)
        {
            return $"{IlOpCode} {string.Join(", ", Operands)}";
        }

        return $"%{Result} = {IlOpCode} {string.Join(", ", Operands)}";
    }
}