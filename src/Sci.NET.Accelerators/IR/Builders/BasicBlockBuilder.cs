// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.IR.Rewriter;

namespace Sci.NET.Accelerators.IR.Builders;

/// <summary>
/// Represents a basic block builder.
/// </summary>
[PublicAPI]
public static class BasicBlockBuilder
{
    /// <summary>
    /// Creates basic blocks from the instructions.
    /// </summary>
    /// <param name="graph">The control flow graph.</param>
    /// <param name="ssaInstructions">The SSA instructions.</param>
    /// <returns>The basic blocks.</returns>
    public static IReadOnlyList<BasicBlock> CreateBasicBlocks(MsilControlFlowGraph graph, ICollection<SsaInstruction> ssaInstructions)
    {
        var basicBlocks = new List<BasicBlock>();

        ConstructBasicBlocks(graph, basicBlocks, ssaInstructions.ToList());
        ConnectBasicBlocks(basicBlocks);
        DetectLoops(basicBlocks);

        return basicBlocks;
    }

    private static void ConnectBasicBlocks(List<BasicBlock> basicBlocks)
    {
        foreach (var basicBlock in basicBlocks)
        {
            foreach (var targetBlock in basicBlock.Instructions[^1].NextInstructions.Select(target => basicBlocks.First(x => x.Instructions[0].Instruction.Offset == target.Offset)))
            {
                basicBlock.NextBlocks.Add(targetBlock);
            }
        }
    }

    private static void ConstructBasicBlocks(MsilControlFlowGraph graph, List<BasicBlock> basicBlocks, IList<SsaInstruction> ssaInstructions)
    {
        var currentNodes = new List<ControlFlowGraphNode>();

        foreach (var node in graph.Nodes)
        {
            if (node.IsLeader && currentNodes.Count != 0)
            {
                var basicBlock = new BasicBlock { StartOffset = currentNodes[0].Instruction.Offset, EndOffset = currentNodes[^1].Instruction.Offset, Instructions = currentNodes };
                basicBlocks.Add(basicBlock);
                currentNodes = new List<ControlFlowGraphNode>
                {
                    new () { Instruction = ssaInstructions[node.Instruction.Index], IsLeader = node.IsLeader, IsTerminator = node.IsTerminator, NextInstructions = node.NextInstructions.Select(x => ssaInstructions[x.Index]).ToList() }
                };
                continue;
            }

            if (node.IsTerminator && currentNodes.Count != 0)
            {
                currentNodes.Add(new ControlFlowGraphNode { Instruction = ssaInstructions[node.Instruction.Index], IsLeader = node.IsLeader, IsTerminator = node.IsTerminator, NextInstructions = node.NextInstructions.Select(x => ssaInstructions[x.Index]).ToList() });
                var basicBlock = new BasicBlock { StartOffset = currentNodes[0].Instruction.Offset, EndOffset = currentNodes[^1].Instruction.Offset, Instructions = currentNodes };
                basicBlocks.Add(basicBlock);
                currentNodes = new List<ControlFlowGraphNode>();
                continue;
            }

            currentNodes.Add(new ControlFlowGraphNode { Instruction = ssaInstructions[node.Instruction.Index], IsLeader = node.IsLeader, IsTerminator = node.IsTerminator, NextInstructions = node.NextInstructions.Select(x => ssaInstructions[x.Index]).ToList() });
        }

        if (currentNodes.Count != 0)
        {
            var basicBlock = new BasicBlock { StartOffset = currentNodes[0].Instruction.Offset, EndOffset = currentNodes[^1].Instruction.Offset, Instructions = currentNodes };
            basicBlocks.Add(basicBlock);
        }
    }

    private static void DetectLoops(List<BasicBlock> basicBlocks)
    {
        foreach (var block in basicBlocks
                     .Where(block => block.NextBlocks.Any(x => x.StartOffset < block.StartOffset)))
        {
            block.IsLoop = true;
        }
    }
}