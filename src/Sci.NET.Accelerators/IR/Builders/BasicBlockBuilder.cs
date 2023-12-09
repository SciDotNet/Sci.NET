// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection.Emit;
using Sci.NET.Accelerators.Disassembly.Instructions;

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
    /// <param name="instructions">The instructions.</param>
    /// <returns>The basic blocks extracted from the instructions.</returns>
    /// <exception cref="InvalidOperationException">Invalid terminator instruction.</exception>
    public static IReadOnlyList<BasicBlock> CreateBasicBlocks(IReadOnlyList<Instruction<IOperand>> instructions)
    {
        var basicBlocks = new List<BasicBlock>();
        var blockEndTargets = new Dictionary<BasicBlock, List<int>>();
        var branchTargets = GetBranchTargets(instructions, basicBlocks, blockEndTargets);

        foreach (var block in blockEndTargets)
        {
            foreach (var target in block.Value)
            {
                if (branchTargets.TryGetValue(target, out var targetBlock))
                {
                    block.Key.NextBlocks.Add(targetBlock);
                }
            }
        }

        for (var i = 0; i < basicBlocks.Count - 1; i++)
        {
            var current = basicBlocks[i];
            var next = basicBlocks[i + 1];

            if (current.NextBlocks.Count == 0 || current.Instructions[^1].IsConditionalBranch)
            {
                current.NextBlocks.Add(next);
            }
        }

        return basicBlocks;
    }

    private static Dictionary<int, BasicBlock> GetBranchTargets(
        IReadOnlyList<Instruction<IOperand>> instructions,
        List<BasicBlock> basicBlocks,
        Dictionary<BasicBlock, List<int>> blockEndTargets)
    {
        var branchTargets = new Dictionary<int, BasicBlock>();
        BasicBlock? currentBlock = null;

        foreach (var instruction in instructions)
        {
            if (currentBlock is null || branchTargets.ContainsKey(instruction.Offset))
            {
                currentBlock = new BasicBlock { StartOffset = instruction.Offset };
                basicBlocks.Add(currentBlock);
                branchTargets[instruction.Offset] = currentBlock;
            }

            currentBlock.Instructions.Add(instruction);

            if (!instruction.IsTerminator)
            {
                continue;
            }

            if (instruction.OpCode.FlowControl == FlowControl.Return)
            {
                continue;
            }

            blockEndTargets[currentBlock] = new List<int>(instruction.GetBranchTargets());

            currentBlock.EndOffset = instruction.Offset;
            currentBlock = null;
        }

        if (currentBlock?.Instructions.Count > 0)
        {
            currentBlock.EndOffset = currentBlock.Instructions[^1].Offset;
        }

        return branchTargets;
    }
}