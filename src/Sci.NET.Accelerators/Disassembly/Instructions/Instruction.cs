// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection.Emit;
using Sci.NET.Accelerators.Disassembly.Instructions.Operands;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly.Instructions;

/// <summary>
/// Represents an instruction with an operand.
/// </summary>
/// <typeparam name="TOperand">The operand type.</typeparam>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public readonly struct Instruction<TOperand> : IInstruction<TOperand>, IValueEquatable<Instruction<TOperand>>
    where TOperand : IOperand
{
    /// <inheritdoc />
    public required OpCode OpCode { get; init; }

    /// <inheritdoc />
    public required int Offset { get; init; }

    /// <inheritdoc />
    public required int Size { get; init; }

    /// <inheritdoc />
    public required TOperand Operand { get; init; }

    /// <inheritdoc />
    public bool IsTerminator => OpCode.FlowControl
        is FlowControl.Branch
        or FlowControl.Cond_Branch
        or FlowControl.Return;

    /// <inheritdoc />
    public bool IsBranch => OpCode.FlowControl
        is FlowControl.Branch
        or FlowControl.Cond_Branch;

    /// <inheritdoc />
    public bool IsUnconditionalBranch => OpCode.FlowControl
        is FlowControl.Branch
        or FlowControl.Return;

    /// <inheritdoc />
    public bool IsConditionalBranch => OpCode.FlowControl
        is FlowControl.Cond_Branch;

    /// <inheritdoc />
    public static bool operator ==(Instruction<TOperand> left, Instruction<TOperand> right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(Instruction<TOperand> left, Instruction<TOperand> right)
    {
        return !(left == right);
    }

    /// <summary>
    /// Gets the branch targets.
    /// </summary>
    /// <returns>The branch targets.</returns>
#pragma warning disable CA1024
    public IEnumerable<int> GetBranchTargets()
#pragma warning restore CA1024
    {
        switch (Operand)
        {
            case BranchTargetOperand branchTarget:
                yield return branchTarget.Target;
                break;
            case SwitchBranchesOperand switchOperand:
                foreach (var target in switchOperand.Branches)
                {
                    yield return target;
                }

                break;
            default:
                yield break;
        }
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(T)" />
    public bool Equals(Instruction<TOperand> other)
    {
        return OpCode.Equals(other.OpCode) &&
               Offset == other.Offset &&
               Size == other.Size &&
               Operand.Equals(other.Operand);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is Instruction<TOperand> other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(
            OpCode,
            Offset,
            Size,
            Operand);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"IL_{Offset} {OpCode} {Operand}";
    }
}