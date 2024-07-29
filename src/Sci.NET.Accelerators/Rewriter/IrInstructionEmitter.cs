// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.Extensions;
using Sci.NET.Accelerators.IR;
using Sci.NET.Accelerators.IR.Instructions;
using Sci.NET.Accelerators.IR.Instructions.MemoryAccess;
using Sci.NET.Accelerators.Rewriter.Variables;

namespace Sci.NET.Accelerators.Rewriter;

internal static class IrInstructionEmitter
{
    public static IInstruction EmitBreak(MsilInstruction<IMsilOperand> node, BasicBlock block, IrGeneratingContext context)
    {
        throw new NotSupportedException("Break is not supported.");
    }

    public static LoadLocalInstruction EmitLdloc(
        MsilInstruction<IMsilOperand> node,
        ISsaVariable result,
        BasicBlock block,
        IrGeneratingContext context)
    {
        if (node.Operand is MsilInlineIntOperand operand)
        {
            return new LoadLocalInstruction
            {
                Local = context.LocalVariableSsaVariables[operand.Value].ToIrValue(),
                Result = result.ToIrValue(),
                MsilInstruction = node,
                Block = block
            };
        }

        if (node.Operand is MsilInlineVarOperand varOperand)
        {
            return new LoadLocalInstruction
            {
                Local = context.LocalVariableSsaVariables[varOperand.Index].ToIrValue(),
                Result = result.ToIrValue(),
                MsilInstruction = node,
                Block = block
            };
        }

        throw new UnreachableException("Ldloc is not supported with the given operands.");
    }

    public static StoreLocalInstruction EmitStloc(
        MsilInstruction<IMsilOperand> node,
        List<ISsaVariable> operands,
        BasicBlock block,
        IrGeneratingContext context)
    {
        if (node.Operand is MsilInlineIntOperand operand)
        {
            return new StoreLocalInstruction
            {
                Local = context.LocalVariableSsaVariables[operand.Value].ToIrValue(),
                Value = operands[0].ToIrValue(),
                MsilInstruction = node,
                Block = block
            };
        }

        if (node.Operand is MsilInlineVarOperand varOperand)
        {
            return new StoreLocalInstruction
            {
                Local = context.LocalVariableSsaVariables[varOperand.Index].ToIrValue(),
                Value = operands[0].ToIrValue(),
                MsilInstruction = node,
                Block = block
            };
        }

        throw new NotSupportedException("Stloc is not supported with the given operands.");
    }

    public static LoadArgumentInstruction EmitLdarg(
        MsilInstruction<IMsilOperand> node,
        ISsaVariable result,
        BasicBlock block,
        IrGeneratingContext context)
    {
        if (node.Operand is MsilInlineIntOperand operand)
        {
            return new LoadArgumentInstruction
            {
                Parameter = context.ArgumentSsaVariables[operand.Value].ToIrValue(),
                Result = result.ToIrValue(),
                MsilInstruction = node,
                Block = block
            };
        }

        if (node.Operand is MsilInlineVarOperand varOperand)
        {
            return new LoadArgumentInstruction
            {
                Parameter = context.ArgumentSsaVariables[varOperand.Index].ToIrValue(),
                Result = result.ToIrValue(),
                MsilInstruction = node,
                Block = block
            };
        }

        throw new NotSupportedException("Ldarg is not supported with the given operands.");
    }

    public static LoadLocalAddressInstruction EmitLdarga(MsilInstruction<IMsilOperand> node, ISsaVariable result, BasicBlock block, IrGeneratingContext context)
    {
        if (node.Operand is MsilInlineIntOperand operand)
        {
            return new LoadLocalAddressInstruction
            {
                Result = result.ToIrValue(),
                Local = context.LocalVariableSsaVariables[operand.Value].ToIrValue(),
                MsilInstruction = node,
                Block = block
            };
        }

        if (node.Operand is MsilInlineVarOperand varOperand)
        {
            return new LoadLocalAddressInstruction
            {
                Result = result.ToIrValue(),
                Local = context.LocalVariableSsaVariables[varOperand.Index].ToIrValue(),
                MsilInstruction = node,
                Block = block
            };
        }

        throw new NotSupportedException("Ldarga is not supported with the given operands.");
    }

    public static IInstruction[] EmitStarg(MsilInstruction<IMsilOperand> node, List<ISsaVariable> operands, BasicBlock block, IrGeneratingContext context)
    {
        var argumentOperand = node.Operand switch
        {
            MsilInlineVarOperand varOperand => context.ArgumentSsaVariables[varOperand.Index],
            MsilInlineIntOperand intOperand => context.ArgumentSsaVariables[intOperand.Value],
            _ => throw new InvalidOperationException("Invalid operand type for starg."),
        };
        var ldargaInstruction = EmitLdarga(node, argumentOperand, block, context);

        var storeInstruction = new StoreElementAtPointerInstruction
        {
            Value = operands[0].ToIrValue(),
            MsilInstruction = node,
            Block = block,
            Pointer = ldargaInstruction.Result
        };

        return new IInstruction[] { ldargaInstruction, storeInstruction };
    }

    public static IInstruction EmitLdnull(MsilInstruction<IMsilOperand> node, ISsaVariable result, BasicBlock block, IrGeneratingContext context)
    {
        throw new NotSupportedException("Ldnull is not supported.");
    }
}