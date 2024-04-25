// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection;
using Sci.NET.Accelerators.Extensions;
using Sci.NET.Accelerators.IR;
using Sci.NET.Accelerators.IR.Instructions;
using Sci.NET.Accelerators.IR.Instructions.MemoryAccess;
using Sci.NET.Accelerators.Rewriter.SpecialInstructions;
using Sci.NET.Common;

namespace Sci.NET.Accelerators.Rewriter.Transforms;

internal class ThreadIndexTransform : IIrTransform
{
    private static MethodInfo GetThreadIdx { get; } =
        typeof(Kernel).GetProperty(nameof(Kernel.ThreadIdx))?.GetMethod ??
        throw new InvalidOperationException();

    private static MethodInfo GetBlockIdx { get; } =
        typeof(Kernel).GetProperty(nameof(Kernel.BlockIdx))?.GetMethod ??
        throw new InvalidOperationException();

    private static MethodInfo GetGridDim { get; } =
        typeof(Kernel).GetProperty(nameof(Kernel.GridDim))?.GetMethod ??
        throw new InvalidOperationException();

    private static MethodInfo GetBlockDim { get; } =
        typeof(Kernel).GetProperty(nameof(Kernel.BlockDim))?.GetMethod ??
        throw new InvalidOperationException();

    private static MethodInfo GetDim3X { get; } =
        typeof(Dim3).GetProperty(nameof(Dim3.X))?.GetMethod ?? throw new InvalidOperationException();

    private static MethodInfo GetDim3Y { get; } =
        typeof(Dim3).GetProperty(nameof(Dim3.Y))?.GetMethod ?? throw new InvalidOperationException();

    private static MethodInfo GetDim3Z { get; } =
        typeof(Dim3).GetProperty(nameof(Dim3.Z))?.GetMethod ?? throw new InvalidOperationException();

    public void Transform(BasicBlock block, ICollection<BasicBlock> allBlocks)
    {
        TransformPass1(block);
        TransformPass2(block, allBlocks);
    }

    private static void TransformPass1(BasicBlock block)
    {
        for (var index = 0; index < block.Instructions.Count; index++)
        {
            if (block.Instructions[index] is not CallInstruction callInstruction)
            {
                continue;
            }

            if (callInstruction.MethodBase == GetThreadIdx)
            {
                block.ReplaceInstruction(
                    index,
                    new LoadDim3ThreadInfoInstruction(callInstruction, Dim3ThreadInformationType.ThreadIdx));
                continue;
            }

            if (callInstruction.MethodBase == GetBlockIdx)
            {
                block.ReplaceInstruction(
                    index,
                    new LoadDim3ThreadInfoInstruction(callInstruction, Dim3ThreadInformationType.BlockIdx));
                continue;
            }

            if (callInstruction.MethodBase == GetGridDim)
            {
                block.ReplaceInstruction(
                    index,
                    new LoadDim3ThreadInfoInstruction(callInstruction, Dim3ThreadInformationType.GridDim));
                continue;
            }

            if (callInstruction.MethodBase == GetBlockDim)
            {
                block.ReplaceInstruction(
                    index,
                    new LoadDim3ThreadInfoInstruction(callInstruction, Dim3ThreadInformationType.BlockDim));
                continue;
            }

            if (callInstruction.MethodBase == GetDim3X)
            {
                block.ReplaceInstruction(index, new LoadDim3ValueInstruction(callInstruction, Dim3Field.X));
                continue;
            }

            if (callInstruction.MethodBase == GetDim3Y)
            {
                block.ReplaceInstruction(index, new LoadDim3ValueInstruction(callInstruction, Dim3Field.Y));
                continue;
            }

            if (callInstruction.MethodBase == GetDim3Z)
            {
                block.ReplaceInstruction(index, new LoadDim3ValueInstruction(callInstruction, Dim3Field.Z));
            }
        }
    }

    private static void TransformPass2(BasicBlock block, ICollection<BasicBlock> allBlocks)
    {
        for (var index = 0; index < block.Instructions.Count; index++)
        {
            var instruction = block.Instructions[index];
            if (instruction is not LoadDim3ValueInstruction dim3Instruction)
            {
                continue;
            }

            var localAddressInstruction = allBlocks.Skip(1).FindDeclaringInstruction(dim3Instruction.Arguments[0]);

            if (localAddressInstruction is not LoadLocalAddressInstruction loadLocalAddressInstruction)
            {
                throw new InvalidOperationException("Could not find declaring instruction.");
            }

            var (storeDim3BlockIndex, storeDim3InstructionIndex) = allBlocks.Skip(1).FindLastStoreInstructionIndex(loadLocalAddressInstruction.Local);

            if (allBlocks.ElementAt(storeDim3BlockIndex + 1).Instructions[storeDim3InstructionIndex] is not StoreLocalInstruction storeDim3Instruction)
            {
                throw new InvalidOperationException("Could not find declaring instruction.");
            }

            var (threadInfoBlockIndex, threadInfoInstructionIndex) = allBlocks.Skip(1).FindDeclaringInstructionIndex(storeDim3Instruction.Value);
            var threadInfoDeclaration = allBlocks.ElementAt(threadInfoBlockIndex + 1).Instructions[threadInfoInstructionIndex];

            if (threadInfoDeclaration is not LoadDim3ThreadInfoInstruction threadInformationInstruction)
            {
                throw new InvalidOperationException("Could not find declaring instruction.");
            }

            block.ReplaceInstruction(
                index,
                new LoadThreadInformationInstruction
                {
                    Result = dim3Instruction.Result,
                    Type =
                        (ThreadInformationType)(((int)threadInformationInstruction.Type + (int)dim3Instruction.Field) << (int)threadInformationInstruction.Type),
                    MsilInstruction = threadInformationInstruction.MsilInstruction
                });

            allBlocks.ElementAt(storeDim3BlockIndex + 1).RemoveInstruction(storeDim3InstructionIndex);
            allBlocks.ElementAt(threadInfoBlockIndex + 1).RemoveInstruction(threadInfoInstructionIndex);
        }
    }
}