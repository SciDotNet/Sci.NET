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
    /// <summary>
    /// Initializes a new instance of the <see cref="BasicBlock"/> class.
    /// </summary>
    /// <param name="name">The name of the basic block.</param>
    /// <param name="msilInstructions">The MSIL instructions of the basic block.</param>
    public BasicBlock(string name, IReadOnlyCollection<MsilInstruction<IMsilOperand>> msilInstructions)
    {
        Name = name;
        Instructions = new List<IInstruction>();
        MsilInstructions = msilInstructions;
    }

    internal BasicBlock(string name, IEnumerable<IInstruction> instructions)
    {
        Name = name;
        Instructions = instructions.ToList();
        MsilInstructions = new List<MsilInstruction<IMsilOperand>>();
    }

    /// <summary>
    /// Gets the instructions of the basic block.
    /// </summary>
    public IReadOnlyList<IInstruction> Instructions { get; private set; }

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
        return WriteToIrString(builder, 0).ToString();
    }

    /// <inheritdoc />
    public StringBuilder WriteToIrString(StringBuilder builder, int indentLevel)
    {
        _ = builder.AppendIndent(indentLevel).Append(Name).Append(':').AppendLine();

        foreach (var instruction in Instructions)
        {
            _ = instruction.WriteToIrString(builder, indentLevel + 1).AppendLine().AppendIndent(indentLevel + 1);
        }

        return builder;
    }

    internal void SetInstructions(IEnumerable<IInstruction> instruction)
    {
        Instructions = instruction.ToList();
    }
}