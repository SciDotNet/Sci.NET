// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Reflection.Emit;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.IR;

namespace Sci.NET.Accelerators.Rewriter;

/// <summary>
/// Builds a new control flow graph.
/// </summary>
[PublicAPI]
public static class CfgBuilder
{
    /// <summary>
    /// Builds a new control flow graph.
    /// </summary>
    /// <param name="method">The disassembled method.</param>
    /// <returns>The new control flow graph.</returns>
    public static ICollection<BasicBlock> Build(DisassembledMsilMethod method)
    {
        var instructions = method.Instructions;
        var leaders = FindLeaders(method, instructions);
        var sortedLeaders = leaders.OrderBy(x => x).ToArray();
        var basicBlocks = FillBasicBlocks(sortedLeaders, instructions);
        var successors = FindSuccessors(basicBlocks, method);

        AddSuccessorsAndPredecessors(successors);

        return basicBlocks;
    }

    private static HashSet<int> FindLeaders(DisassembledMsilMethod method, ImmutableArray<MsilInstruction<IMsilOperand>> instructions)
    {
        var leaders = new HashSet<int> { 0 };

        for (var i = 0; i < instructions.Length; i++)
        {
            var instruction = instructions[i];
            if (instruction.FlowControl is FlowControl.Branch or FlowControl.Cond_Branch)
            {
                _ = leaders.Add(i + 1);

                foreach (var target in instruction.GetBranchTargetInstructionIndices(method))
                {
                    _ = leaders.Add(target);
                }
            }

            if (instruction.FlowControl is FlowControl.Return)
            {
                _ = leaders.Add(i + 1);
            }

            if (instruction.FlowControl is FlowControl.Throw)
            {
                _ = leaders.Add(i + 1);
            }
        }

        return leaders;
    }

    private static List<BasicBlock> FillBasicBlocks(int[] sortedLeaders, ImmutableArray<MsilInstruction<IMsilOperand>> instructions)
    {
        var basicBlocks = new List<BasicBlock>();

        for (var i = 0; i < sortedLeaders.Length - 1; i++)
        {
            var leader = sortedLeaders[i];
            var terminator = sortedLeaders[i + 1];
            var block = new List<MsilInstruction<IMsilOperand>>();

            for (var j = leader; j < terminator; j++)
            {
                block.Add(instructions[j]);
            }

            if (block.Count == 0)
            {
                continue;
            }

            basicBlocks.Add(new BasicBlock($"block_{i + 1}", block));
        }

        return basicBlocks;
    }

    private static Dictionary<BasicBlock, List<BasicBlock>> FindSuccessors(List<BasicBlock> basicBlocks, DisassembledMsilMethod method)
    {
        var successors = new Dictionary<BasicBlock, List<BasicBlock>>();

        foreach (var block in basicBlocks)
        {
            var lastInstruction = block.MsilInstructions.Last();

            if (lastInstruction.FlowControl is FlowControl.Branch or FlowControl.Cond_Branch)
            {
                foreach (var targetIndex in lastInstruction.GetBranchTargetInstructionIndices(method))
                {
                    var targetBlock = basicBlocks.Find(x => x.MsilInstructions.First().Index == targetIndex);

                    if (targetBlock is not null)
                    {
                        if (!successors.TryGetValue(block, out var value))
                        {
                            value ??= new List<BasicBlock>();
                            successors[block] = value;
                        }

                        value.Add(targetBlock);
                    }
                }
            }
        }

        return successors;
    }

    private static void AddSuccessorsAndPredecessors(Dictionary<BasicBlock, List<BasicBlock>> successorMapping)
    {
        foreach (var (block, successors) in successorMapping)
        {
            foreach (var successor in successors)
            {
                block.AddSuccessor(successor);
                successor.AddPredecessor(block);
            }
        }
    }
}