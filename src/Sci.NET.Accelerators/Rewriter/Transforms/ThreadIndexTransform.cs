// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection;
using Sci.NET.Accelerators.IR;
using Sci.NET.Accelerators.IR.Instructions;
using Sci.NET.Accelerators.Rewriter.SpecialInstructions;
using Sci.NET.Common;

namespace Sci.NET.Accelerators.Rewriter.Transforms;

internal class ThreadIndexTransform : IInstructionTransform
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

    public void Transform(BasicBlock block)
    {
        TransformPass1(block);
        TransformPass2(block);
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
                block.ReplaceInstruction(index, new LoadDim3ThreadInfoInstruction(callInstruction, ThreadInfoType.ThreadIdx));
                continue;
            }

            if (callInstruction.MethodBase == GetBlockIdx)
            {
                block.ReplaceInstruction(index, new LoadDim3ThreadInfoInstruction(callInstruction, ThreadInfoType.BlockIdx));
                continue;
            }

            if (callInstruction.MethodBase == GetGridDim)
            {
                block.ReplaceInstruction(index, new LoadDim3ThreadInfoInstruction(callInstruction, ThreadInfoType.GridDim));
                continue;
            }

            if (callInstruction.MethodBase == GetBlockDim)
            {
                block.ReplaceInstruction(index, new LoadDim3ThreadInfoInstruction(callInstruction, ThreadInfoType.BlockDim));
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

    private static void TransformPass2(BasicBlock block)
    {
        _ = block;
    }
}