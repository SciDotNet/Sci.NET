// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection.Emit;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.Disassembly.Pdb;
using Sci.NET.Common.Comparison;

namespace Sci.NET.Accelerators.Disassembly;

/// <summary>
/// Represents a disassembled MSIL instruction.
/// </summary>
/// <typeparam name="TOperand">The type of the operand.</typeparam>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop doesnt support required members yet.")]
public readonly struct MsilInstruction<TOperand> : IValueEquatable<MsilInstruction<TOperand>>
    where TOperand : IMsilOperand
{
     /// <summary>
    /// Initializes a new instance of the <see cref="MsilInstruction{TOperand}"/> struct.
    /// </summary>
    /// <param name="opcode">The opcode.</param>
    /// <param name="offset">The offset.</param>
    /// <param name="size">The size.</param>
    /// <param name="index">The index of the instruction.</param>
    [SetsRequiredMembers]
    public MsilInstruction(OpCode opcode, int offset, int size, int index)
    {
        IlOpCode = opcode.Type();
        PopBehaviour = (PopBehaviour)opcode.StackBehaviourPop;
        PushBehaviour = (PushBehaviour)opcode.StackBehaviourPush;
        FlowControl = opcode.FlowControl;
        Offset = offset;
        Size = size;
        OperandType = opcode.OperandType;
        Name = opcode.Name ?? string.Empty;
        Operand = default!;
        Index = index;
    }

    /// <summary>
    ///  Gets the instruction name.
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Gets the instruction opcode.
    /// </summary>
    public required OpCodeTypes IlOpCode { get; init; }

    /// <summary>
    /// Gets the instruction pop behaviour.
    /// </summary>
    public required PopBehaviour PopBehaviour { get; init; }

    /// <summary>
    /// Gets the instruction push behaviour.
    /// </summary>
    public required PushBehaviour PushBehaviour { get; init; }

    /// <summary>
    /// Gets the instruction flow control.
    /// </summary>
    public required FlowControl FlowControl { get; init; }

    /// <summary>
    /// Gets the operand type.
    /// </summary>
    public required OperandType OperandType { get; init; }

    /// <summary>
    /// Gets the instruction offset.
    /// </summary>
    public required int Offset { get; init; }

    /// <summary>
    /// Gets the instruction size.
    /// </summary>
    public required int Size { get; init; }

    /// <summary>
    /// Gets the instruction operand.
    /// </summary>
    public required TOperand Operand { get; init; }

    /// <summary>
    /// Gets the index of the instruction.
    /// </summary>
    public required int Index { get; init; }

    /// <summary>
    /// Gets a value indicating whether the instruction is a terminator.
    /// </summary>
    public bool IsTerminator => FlowControl
        is FlowControl.Branch
        or FlowControl.Cond_Branch
        or FlowControl.Return;

    /// <summary>
    /// Gets a value indicating whether the instruction is a branch.
    /// </summary>
    public bool IsBranch => FlowControl
        is FlowControl.Branch
        or FlowControl.Cond_Branch;

    /// <summary>
    ///  Gets a value indicating whether the instruction is an unconditional branch.
    /// </summary>
    public bool IsUnconditionalBranch => FlowControl
        is FlowControl.Branch
        or FlowControl.Return;

    /// <summary>
    /// Gets a value indicating whether the instruction is a conditional branch.
    /// </summary>
    public bool IsConditionalBranch => FlowControl
        is FlowControl.Cond_Branch;

    /// <summary>
    /// Gets the sequence point.
    /// </summary>
    public PdbSequencePoint? SequencePoint { get; init; }

    /// <summary>
    /// Gets a value indicating whether the instruction is a call instruction.
    /// </summary>
    public bool IsCall => IlOpCode is OpCodeTypes.Call or OpCodeTypes.Callvirt or OpCodeTypes.Calli;

    /// <inheritdoc />
    public static bool operator ==(MsilInstruction<TOperand> left, MsilInstruction<TOperand> right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc />
    public static bool operator !=(MsilInstruction<TOperand> left, MsilInstruction<TOperand> right)
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
            case MsilBranchTargetOperand branchTarget:
                yield return branchTarget.Target;
                break;
            case MsilSwitchTargetsOperand switchOperand:
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
    public bool Equals(MsilInstruction<TOperand> other)
    {
        return IlOpCode.Equals(other.IlOpCode) &&
               Offset == other.Offset &&
               Size == other.Size &&
               Operand.Equals(other.Operand);
    }

    /// <inheritdoc cref="IValueEquatable{T}.Equals(object?)" />
    public override bool Equals(object? obj)
    {
        return obj is MsilInstruction<TOperand> other && Equals(other);
    }

    /// <inheritdoc cref="IValueEquatable{T}.GetHashCode" />
    public override int GetHashCode()
    {
        return HashCode.Combine(
            IlOpCode,
            Offset,
            Size,
            Operand);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"IL_{Offset:x4} {IlOpCode} {Operand}";
    }
}