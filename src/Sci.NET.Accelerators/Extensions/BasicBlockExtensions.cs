// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Text;
using Sci.NET.Accelerators.IR;
using Sci.NET.Accelerators.IR.Instructions;
using Sci.NET.Accelerators.IR.Instructions.MemoryAccess;

namespace Sci.NET.Accelerators.Extensions;

/// <summary>
/// Extension methods for <see cref="BasicBlock"/>.
/// </summary>
[PublicAPI]
public static class BasicBlockExtensions
{
    /// <summary>
    /// Converts a collection <see cref="BasicBlock"/> to an IR string.
    /// </summary>
    /// <param name="basicBlocks">The collection of <see cref="BasicBlock"/> to convert.</param>
    /// <returns>The IR string representation of the <see cref="BasicBlock"/>s.</returns>
    public static string ToIR(this IEnumerable<BasicBlock> basicBlocks)
    {
        var sb = new StringBuilder();

        foreach (var block in basicBlocks)
        {
            _ = block.WriteToIrString(sb);
        }

        return sb.ToString();
    }

    /// <summary>
    /// Finds the instruction that declares the given value.
    /// </summary>
    /// <param name="blocks">The basic block to search.</param>
    /// <param name="value">The value to find the declaring instruction for.</param>
    /// <returns>The instruction that declares the given value.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the declaring instruction could not be found.</exception>
    public static IValueYieldingInstruction FindDeclaringInstruction(this IEnumerable<BasicBlock> blocks, IrValue value)
    {
        var basicBlocks = blocks as BasicBlock[] ?? blocks.ToArray();
        var (blockIndex, instructionIndex) = basicBlocks.FindDeclaringInstructionIndex(value);

        return basicBlocks[blockIndex].Instructions[instructionIndex] as IValueYieldingInstruction
               ?? throw new InvalidOperationException("Could not find declaring instruction.");
    }

    /// <summary>
    /// Finds the instruction that declares the given value.
    /// </summary>
    /// <param name="blocks">The basic block to search.</param>
    /// <param name="value">The value to find the declaring instruction for.</param>
    /// <returns>The instruction that declares the given value.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the declaring instruction could not be found.</exception>
    public static (int BlockIndex, int InstructionIndex) FindDeclaringInstructionIndex(this IEnumerable<BasicBlock> blocks, IrValue value)
    {
        var blocksArray = blocks.ToArray();

        for (var blockIdx = 0; blockIdx < blocksArray.Length; blockIdx++)
        {
            var block = blocksArray[blockIdx];
            for (var instructionIdx = 0; instructionIdx < block.Instructions.Count; instructionIdx++)
            {
                var instruction = block.Instructions[instructionIdx];
                if (instruction is not IValueYieldingInstruction yieldingInstruction)
                {
                    continue;
                }

                if (yieldingInstruction.Result == value)
                {
                    return (blockIdx, instructionIdx);
                }
            }
        }

        throw new InvalidOperationException("Could not find declaring instruction.");
    }

    /// <summary>
    /// Finds the last store instruction for the given value.
    /// </summary>
    /// <param name="blocks">The basic blocks to search.</param>
    /// <param name="value">The value to find the last store instruction for.</param>
    /// <returns>The last store instruction for the given value.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the store instruction could not be found.</exception>
    public static StoreLocalInstruction FindLastStoreInstruction(this IEnumerable<BasicBlock> blocks, IrValue value)
    {
        var basicBlocks = blocks as BasicBlock[] ?? blocks.ToArray();
        var (blockIdx, index) = basicBlocks.FindLastStoreInstructionIndex(value);

        return basicBlocks[blockIdx].Instructions[index] as StoreLocalInstruction
               ?? throw new InvalidOperationException("Could not find declaring instruction.");
    }

    /// <summary>
    /// Finds the last store instruction for the given value.
    /// </summary>
    /// <param name="blocks">The basic blocks to search.</param>
    /// <param name="value">The value to find the last store instruction for.</param>
    /// <exception cref="InvalidOperationException">Thrown if the store instruction could not be found.</exception>
    /// <returns>The index of the last store instruction for the given value.</returns>
    public static (int BlockIndex, int InstructionIndex) FindLastStoreInstructionIndex(this IEnumerable<BasicBlock> blocks, IrValue value)
    {
        var blocksArray = blocks.ToArray();
        for (var blockIdx = 0; blockIdx < blocksArray.Length; blockIdx++)
        {
            var block = blocksArray[blockIdx];
            for (var index = 0; index < block.Instructions.Count; index++)
            {
                var instruction = block.Instructions[index];
                if (instruction is not StoreLocalInstruction storeLocalInstruction)
                {
                    continue;
                }

                if (storeLocalInstruction.Local == value)
                {
                    return (blockIdx, index);
                }
            }
        }

        throw new InvalidOperationException("Could not find declaring instruction.");
    }
}