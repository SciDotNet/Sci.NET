// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Text;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.Extensions;
using Sci.NET.Accelerators.IR.Instructions;

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// A basic block.
/// </summary>
[PublicAPI]
public class BasicBlock : IIrWritable
{
    private readonly List<IInstruction> _instructions;

    /// <summary>
    /// Initializes a new instance of the <see cref="BasicBlock"/> class.
    /// </summary>
    /// <param name="name">The name of the basic block.</param>
    /// <param name="msilInstructions">The MSIL instructions of the basic block.</param>
    public BasicBlock(string name, IReadOnlyCollection<MsilInstruction<IMsilOperand>> msilInstructions)
    {
        Name = name;
        MsilInstructions = msilInstructions;
        _instructions = new List<IInstruction>();
    }

    internal BasicBlock(string name, IEnumerable<IInstruction> instructions)
    {
        Name = name;
        MsilInstructions = new List<MsilInstruction<IMsilOperand>>();
        _instructions = instructions.ToList();
    }

    /// <summary>
    /// Gets the instructions of the basic block.
    /// </summary>
    public IReadOnlyList<IInstruction> Instructions => _instructions;

    /// <summary>
    /// Gets the MSIL instructions of the basic block.
    /// </summary>
    public IReadOnlyCollection<MsilInstruction<IMsilOperand>> MsilInstructions { get; init; }

    /// <summary>
    /// Gets the name of the basic block.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets a value indicating whether the basic block is a leader for the given offset.
    /// </summary>
    /// <param name="offset">The offset.</param>
    /// <returns><c>true</c> if the basic block is a leader for the given offset, otherwise <c>false</c>.</returns>
    public bool IsLeaderFor(int offset)
    {
        return MsilInstructions.Any(x => x.Offset == offset);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var builder = new StringBuilder();
        return WriteToIrString(builder).ToString();
    }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder)
    {
        _ = builder.AppendIndent(0).Append(Name).Append(':').AppendLine();

        foreach (var instruction in Instructions)
        {
            if (instruction is NopInstruction)
            {
                continue;
            }

            _ = builder.AppendIndent(1);
            _ = instruction.WriteToIrString(builder).AppendLine();
        }

        return builder;
    }

    /// <summary>
    /// Sets the instructions of the basic block.
    /// </summary>
    /// <param name="instruction">The instructions to set.</param>
    public void SetInstructions(IEnumerable<IInstruction> instruction)
    {
        _instructions.Clear();
        _instructions.AddRange(instruction.ToList());
    }

    /// <summary>
    /// Replaces the instruction at the given index with the given instruction.
    /// </summary>
    /// <param name="index">The index of the instruction to replace.</param>
    /// <param name="instruction">The instruction to replace with.</param>
    public void ReplaceInstruction(int index, IInstruction instruction)
    {
        _instructions[index] = instruction;
    }

    /// <summary>
    /// Removes the instruction at the given index.
    /// </summary>
    /// <param name="index">The index of the instruction to remove.</param>
    public void RemoveInstruction(int index)
    {
        _instructions.RemoveAt(index);
    }
}