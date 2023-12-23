// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection.Emit;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;

namespace Sci.NET.Accelerators.IR.Rewriter;

/// <summary>
/// Represents a control flow graph.
/// </summary>
[PublicAPI]
public sealed class MsilControlFlowGraph
{
    private MsilControlFlowGraph(List<IMsilControlFlowGraphNode> nodes, List<Instruction<IOperand>> leaders)
    {
        Nodes = nodes;
        Leaders = leaders;
    }

    /// <summary>
    /// Gets the leaders of the control flow graph.
    /// </summary>
    public IEnumerable<Instruction<IOperand>> Leaders { get; }

    /// <summary>
    /// Gets the nodes of the control flow graph.
    /// </summary>
    public IList<IMsilControlFlowGraphNode> Nodes { get; }

    /// <summary>
    /// Creates a control flow graph from the instructions.
    /// </summary>
    /// <param name="instructions">The instructions.</param>
    /// <returns>The control flow graph.</returns>
    public static MsilControlFlowGraph Create(IList<Instruction<IOperand>> instructions)
    {
        var nodes = new List<MsilControlFlowGraphNode>();
        var instructionsList = instructions.ToList();
        var leaders = new List<Instruction<IOperand>> { instructionsList[0] };
        var terminators = instructionsList.Where(x => x.FlowControl is FlowControl.Branch or FlowControl.Cond_Branch).ToList();

        for (var i = 0; i < instructionsList.Count; i++)
        {
            var instruction = instructionsList[i];
            var nextInstructions = new List<Instruction<IOperand>>();

#pragma warning disable IDE0010
            switch (instruction.Operand)
#pragma warning restore IDE0010
            {
                case BranchTargetOperand operand:
                    var target = instructionsList.First(x => x.Offset == operand.Target);

                    nextInstructions.Add(target);
                    leaders.Add(target);
                    terminators.Add(instruction);

                    if (instruction.FlowControl == FlowControl.Cond_Branch)
                    {
                        nextInstructions.Add(instructionsList[i + 1]);
                        leaders.Add(instructionsList[i + 1]);
                    }

                    break;

                case SwitchBranchesOperand switchBranchesOperand:
                    var branches = switchBranchesOperand
                        .Branches
                        .Select(branch => instructionsList.First(x => x.Offset == branch))
                        .ToArray();

                    nextInstructions.AddRange(branches);
                    leaders.AddRange(branches);
                    terminators.Add(instruction);
                    break;
                default:
                    if (instruction.FlowControl == FlowControl.Return)
                    {
                        terminators.Add(instruction);
                        break;
                    }

                    nextInstructions.Add(instructionsList[i + 1]);
                    break;
            }

            nodes.Add(new MsilControlFlowGraphNode(instruction, nextInstructions));
        }

        foreach (var leader in leaders)
        {
            nodes.First(x => x.Instruction.Offset == leader.Offset).IsLeader = true;
        }

        foreach (var terminator in terminators)
        {
            nodes.First(x => x.Instruction.Offset == terminator.Offset).IsTerminator = true;
        }

        return new MsilControlFlowGraph([..nodes], leaders);
    }
}