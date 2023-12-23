// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

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
    /// <returns>The basic blocks.</returns>
    public static IReadOnlyList<BasicBlock> CreateBasicBlocks(ControlFlowGraph graph)
    {
        var basicBlocks = new List<BasicBlock>();

        ConstructBasicBlocks(graph, basicBlocks);
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

    private static void ConstructBasicBlocks(ControlFlowGraph graph, List<BasicBlock> basicBlocks)
    {
        var currentNodes = new List<IControlFlowGraphNode>();

        foreach (var node in graph.Nodes)
        {
            if (node.IsLeader && currentNodes.Count != 0)
            {
                var basicBlock = new BasicBlock { StartOffset = currentNodes[0].Instruction.Offset, EndOffset = currentNodes[^1].Instruction.Offset, Instructions = currentNodes };
                basicBlocks.Add(basicBlock);
                currentNodes = new List<IControlFlowGraphNode>();
                currentNodes.Add(node);
                continue;
            }

            if (node.IsTerminator && currentNodes.Count != 0)
            {
                currentNodes.Add(node);
                var basicBlock = new BasicBlock { StartOffset = currentNodes[0].Instruction.Offset, EndOffset = currentNodes[^1].Instruction.Offset, Instructions = currentNodes };
                basicBlocks.Add(basicBlock);
                currentNodes = new List<IControlFlowGraphNode>();
                continue;
            }

            currentNodes.Add(node);
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