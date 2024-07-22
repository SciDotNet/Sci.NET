// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Reflection;
using System.Reflection.Emit;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.Extensions;
using Sci.NET.Accelerators.IR;
using Sci.NET.Accelerators.IR.Instructions;
using Sci.NET.Accelerators.IR.Instructions.Arithmetic;
using Sci.NET.Accelerators.IR.Instructions.Comparison;
using Sci.NET.Accelerators.IR.Instructions.Conversion;
using Sci.NET.Accelerators.IR.Instructions.MemoryAccess;
using Sci.NET.Accelerators.IR.Instructions.Terminators;
using Sci.NET.Accelerators.Rewriter.Transforms;
using Sci.NET.Accelerators.Rewriter.Variables;

namespace Sci.NET.Accelerators.Rewriter;

/// <summary>
/// Converts from the stack-based MSIL intermediate representation to an SSA based representation.
/// </summary>
[PublicAPI]
public class MsilToIrTranslator
{
#pragma warning disable SA1010, SA1009
    private static readonly Type[] TypePrecedenceBitManipulation =
    [
        typeof(ulong), typeof(long), typeof(uint), typeof(int), typeof(ushort), typeof(short), typeof(byte),
        typeof(sbyte)
    ];

    private static readonly Type[] TypePrecedenceArithmetic =
    [
        typeof(nuint), typeof(nint), typeof(double), typeof(float), typeof(ulong), typeof(long), typeof(uint),
        typeof(int), typeof(ushort), typeof(short), typeof(byte), typeof(sbyte)
    ];
#pragma warning restore SA1009, SA1010

    private readonly IrGeneratingContext _context;

    /// <summary>
    /// Initializes a new instance of the <see cref="MsilToIrTranslator"/> class.
    /// </summary>
    /// <param name="disassembledMethod">The disassembled method.</param>
    public MsilToIrTranslator(DisassembledMsilMethod disassembledMethod)
    {
        _context = new IrGeneratingContext(disassembledMethod);
    }

    /// <summary>
    /// Transforms the method into SSA form.
    /// </summary>
    /// <param name="blocks">The basic blocks of the method.</param>
    /// <returns>The SSA form of the method.</returns>
    public MsilSsaMethod Transform(ICollection<BasicBlock> blocks)
    {
        var basicBlocksList = blocks.ToList();
        basicBlocksList.Insert(0, GetLocalsInitBlock(basicBlocksList[0]));

        var basicBlocks = basicBlocksList.ToArray();

        for (var blockIdx = 1; blockIdx < basicBlocks.Length; blockIdx++)
        {
            var block = basicBlocks[blockIdx];
            var blockInstructions = new List<IInstruction>();

            foreach (var node in block.MsilInstructions)
            {
                var operands = ExtractOperands(node).ToList();

                var popCount = GetPopCount(node);
                var poppedItems = new ISsaVariable[popCount];
                var count = popCount;

                while (count > 0)
                {
                    poppedItems[count - 1] = _context.Stack.Pop();
                    count--;
                }

                for (var i = 0; i < popCount; i++)
                {
                    operands.Add(poppedItems[i]);
                }

                var result = GetResult(node, operands);

                if (result.Type != typeof(void))
                {
                    _context.Stack.Push(result);
                }

                var symbol = GetInstruction(
                    node,
                    operands,
                    result,
                    block,
                    blockIdx == basicBlocks.Length - 1 ? basicBlocks[blockIdx] : basicBlocks[blockIdx + 1],
                    basicBlocks);

                blockInstructions.Add(symbol);
            }

            if (blockInstructions[^1].MsilInstruction?.FlowControl is
                    not FlowControl.Branch and not FlowControl.Cond_Branch and not FlowControl.Return &&
                blockIdx < basicBlocks.Length - 1)
            {
                blockInstructions.Add(
                    new BranchInstruction
                    {
                        Target = basicBlocks[blockIdx + 1],
                        MsilInstruction = null,
                        Block = block
                    });
            }

            block.SetInstructions(blockInstructions);
        }

        foreach (var transform in TransformManager.GetTransforms())
        {
            foreach (var basicBlock in basicBlocks)
            {
                transform.Transform(basicBlock, basicBlocks);
            }
        }

        new DeadBlockRemover().Transform(basicBlocks);

        return new MsilSsaMethod
        {
            BasicBlocks = basicBlocks,
            Locals = _context.LocalVariableSsaVariables,
            Parameters = _context.ArgumentSsaVariables,
            ReturnType = _context.DisassembledMethod.Metadata.ReturnType,
            Metadata = _context.DisassembledMethod.Metadata
        };
    }

#pragma warning disable RCS1213
    private static BasicBlock FindBlockForOffset(BasicBlock[] basicBlocks, int offset)
#pragma warning restore RCS1213
    {
        foreach (var basicBlock in basicBlocks)
        {
            if (basicBlock.IsLeaderFor(offset))
            {
                return basicBlock;
            }
        }

        throw new InvalidOperationException("The offset does not belong to any basic block.");
    }

    private static Type GetResultTypeFromBitManipulationOperation(IEnumerable<ISsaVariable> operands)
    {
        var operandsArray = operands as ISsaVariable[] ?? operands.ToArray();

        if (operandsArray.Length != 2)
        {
            throw new InvalidOperationException("Invalid number of operands for binary operation.");
        }

        var first = operandsArray[0];
        var second = operandsArray[^1];

        foreach (var type in TypePrecedenceBitManipulation)
        {
            if (HasType(type, first, second))
            {
                return type;
            }
        }

        throw new InvalidOperationException("Invalid operand types for binary operation.");
    }

    private static Type GetResultTypeFromBinaryArithmeticOperation(IEnumerable<ISsaVariable> operands)
    {
        var operandsArray = operands as ISsaVariable[] ?? operands.ToArray();

        if (operandsArray.Length != 2)
        {
            throw new InvalidOperationException("Invalid number of operands for binary operation.");
        }

        var first = operandsArray[0];
        var second = operandsArray[^1];

        if (first.Type.IsPointer || second.Type.IsPointer)
        {
            if (first.Type == second.Type)
            {
                return first.Type;
            }

            if (first.Type == typeof(nint) || first.Type == typeof(nuint))
            {
                return second.Type;
            }

            if (second.Type == typeof(nint) || second.Type == typeof(nuint))
            {
                return first.Type;
            }

            throw new InvalidOperationException("Invalid operand types for binary operation.");
        }

        foreach (var type in TypePrecedenceArithmetic)
        {
            if (HasType(type, first, second))
            {
                return type;
            }
        }

        throw new InvalidOperationException("Invalid operand types for binary operation.");
    }

    private static bool HasType(Type type, ISsaVariable first, ISsaVariable second)
    {
        return first.Type == type || second.Type == type;
    }

    private static int GetMethodCallPopBehaviour(MsilMethodOperand operand)
    {
        var method = operand.MethodBase;

        if (method is not MethodInfo methodInfo)
        {
            throw new InvalidOperationException("The method operand does not have a method base.");
        }

        var popCount = methodInfo.GetParameters().Length;

#pragma warning disable IDE0010
        switch (methodInfo.CallingConvention & (CallingConventions.VarArgs | CallingConventions.HasThis))
#pragma warning restore IDE0010
        {
            case CallingConventions.VarArgs:
                throw new NotSupportedException(
                    "Calling methods with variable number of arguments is not yet supported.");
            case CallingConventions.HasThis:
                popCount++;
                break;
        }

        return popCount;
    }

    private BasicBlock GetLocalsInitBlock(BasicBlock firstCodeBlock)
    {
        var locals = new List<IInstruction>();

        foreach (var local in _context.LocalVariableSsaVariables)
        {
            var irLocal = local.ToIrValue();
            locals.Add(
                new DeclareLocalInstruction
                {
                    Result = local.ToIrValue(),
                    Value = irLocal.Type.CreateDefaultInstance(),
                    MsilInstruction = null,
                    Block = firstCodeBlock
                });
        }

        locals.Add(
            new BranchInstruction
            {
                Target = firstCodeBlock,
                MsilInstruction = null,
                Block = firstCodeBlock
            });

        var firstBlock = new BasicBlock("block_0", locals);
        firstBlock.AddSuccessor(firstCodeBlock);
        firstCodeBlock.AddPredecessor(firstBlock);

        return firstBlock;
    }

    private List<ISsaVariable> ExtractOperands(MsilInstruction<IMsilOperand> node)
    {
        var operands = new List<ISsaVariable>();
#pragma warning disable IDE0010
        switch (node.IlOpCode)
#pragma warning restore IDE0010
        {
            case OpCodeTypes.Ldarg or OpCodeTypes.Ldarg_S when node.Operand is MsilInlineIntOperand index:
                operands.Add(_context.ArgumentSsaVariables[index.Value]);
                break;
            case OpCodeTypes.Ldloc or OpCodeTypes.Ldloc_S when node.Operand is MsilInlineIntOperand index:
                operands.Add(_context.LocalVariableSsaVariables[index.Value]);
                break;
            case OpCodeTypes.Ldc_I4 or OpCodeTypes.Ldc_I4_S when node.Operand is MsilInlineIntOperand value:
                operands.Add(new Int4ConstantSsaVariable(_context.NameGenerator.NextTemporaryName(), value.Value));
                break;
            case OpCodeTypes.Ldc_I8 when node.Operand is MsilInlineLongOperand value:
                operands.Add(new Int8ConstantSsaVariable(_context.NameGenerator.NextTemporaryName(), value.Value));
                break;
            case OpCodeTypes.Ldc_R4 when node.Operand is MsilInlineSingleOperand value:
                operands.Add(new Float4ConstantSsaVariable(_context.NameGenerator.NextTemporaryName(), value.Value));
                break;
            case OpCodeTypes.Ldc_R8 when node.Operand is MsilInlineDoubleOperand value:
                operands.Add(new Float8ConstantSsaVariable(_context.NameGenerator.NextTemporaryName(), value.Value));
                break;
        }

        return operands;
    }

    private int GetPopCount(MsilInstruction<IMsilOperand> node)
    {
        return node.PopBehaviour switch
        {
            PopBehaviour.None => 0,
            PopBehaviour.Pop1 or PopBehaviour.Popi or PopBehaviour.Popref => 1,
            PopBehaviour.Pop1_pop1 or PopBehaviour.Popi_popi or PopBehaviour.Popi_popi8 or PopBehaviour.Popi_popr4
                or PopBehaviour.Popi_popr8 or PopBehaviour.Popref_pop1 or PopBehaviour.Popi_pop1
                or PopBehaviour.Popref_popi => 2,
            PopBehaviour.Popi_popi_popi or PopBehaviour.Popref_popi_popi or PopBehaviour.Popref_popi_popi8
                or PopBehaviour.Popref_popi_popr4 or PopBehaviour.Popref_popi_popr8 or PopBehaviour.Popref_popi_popref
                or PopBehaviour.Popref_popi_pop1 => 3,
            PopBehaviour.Varpop when node.IlOpCode is OpCodeTypes.Ret &&
                                     _context.DisassembledMethod.Metadata.ReturnType != typeof(void) => 0,
            PopBehaviour.Varpop when node.IlOpCode is OpCodeTypes.Ret && _context.Stack.Count == 0 &&
                                     _context.DisassembledMethod.Metadata.ReturnType == typeof(void) => 0,
            PopBehaviour.Varpop when node.IlOpCode is OpCodeTypes.Ret && _context.Stack.Count == 1 => 1,
            PopBehaviour.Varpop when node.IlOpCode is OpCodeTypes.Ret && _context.Stack.Count > 0 =>
                throw new InvalidOperationException("The stack is not empty."),
            PopBehaviour.Varpop when node.Operand is MsilMethodOperand operand => GetMethodCallPopBehaviour(operand),
            PopBehaviour.Varpop => throw new NotSupportedException("Varpop is not supported."),
            _ => throw new InvalidOperationException("The pop behaviour is not supported.")
        };
    }

    private ISsaVariable GetResult(MsilInstruction<IMsilOperand> node, List<ISsaVariable> operands)
    {
        // This code was generated procedurally from documentation. It is verified to be correct
        // but we should be ignoring short form (inst_s) and macro (inst_m) instructions.
        // This can be an improvement for the future as it will (marginally) improve performance.
#pragma warning disable IDE0072
        return node.IlOpCode switch
#pragma warning restore IDE0072
        {
            OpCodeTypes.Nop => default(VoidSsaVariable),
            OpCodeTypes.Break => default(VoidSsaVariable),
            OpCodeTypes.Ldnull => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldc_I4 or OpCodeTypes.Ldc_I4_S => _context.NameGenerator.GetNextTemp(typeof(int)),
            OpCodeTypes.Ldc_I8 => _context.NameGenerator.GetNextTemp(typeof(long)),
            OpCodeTypes.Ldc_R4 => _context.NameGenerator.GetNextTemp(typeof(float)),
            OpCodeTypes.Ldc_R8 => _context.NameGenerator.GetNextTemp(typeof(double)),
            OpCodeTypes.Dup => operands[0],
            OpCodeTypes.Pop => _context.Stack.Peek(),
            OpCodeTypes.Jmp => default(VoidSsaVariable),
            OpCodeTypes.Call or OpCodeTypes.Callvirt => GetMethodCallReturnType(
                node.Operand is MsilMethodOperand operand ? operand : default),
            OpCodeTypes.Calli => GetMethodCallReturnType(node.Operand is MsilMethodOperand operand ? operand : default),
            OpCodeTypes.Ret => default(VoidSsaVariable),
            OpCodeTypes.Br_S
                or OpCodeTypes.Brfalse_S
                or OpCodeTypes.Brtrue_S
                or OpCodeTypes.Beq_S
                or OpCodeTypes.Bge_S
                or OpCodeTypes.Bgt_S
                or OpCodeTypes.Ble_S
                or OpCodeTypes.Blt_S
                or OpCodeTypes.Bne_Un_S
                or OpCodeTypes.Bge_Un_S
                or OpCodeTypes.Bgt_Un_S
                or OpCodeTypes.Ble_Un_S
                or OpCodeTypes.Blt_Un_S
                or OpCodeTypes.Leave_S
                or OpCodeTypes.Br
                or OpCodeTypes.Brfalse
                or OpCodeTypes.Brtrue
                or OpCodeTypes.Beq
                or OpCodeTypes.Bge
                or OpCodeTypes.Bgt
                or OpCodeTypes.Ble
                or OpCodeTypes.Blt
                or OpCodeTypes.Bne_Un
                or OpCodeTypes.Bge_Un
                or OpCodeTypes.Bgt_Un
                or OpCodeTypes.Ble_Un
                or OpCodeTypes.Blt_Un => default(VoidSsaVariable),
            OpCodeTypes.Switch => default(VoidSsaVariable),
            OpCodeTypes.Ldind_I1 => _context.NameGenerator.GetNextTemp(typeof(sbyte)),
            OpCodeTypes.Ldind_U1 => _context.NameGenerator.GetNextTemp(typeof(byte)),
            OpCodeTypes.Ldind_I2 => _context.NameGenerator.GetNextTemp(typeof(short)),
            OpCodeTypes.Ldind_U2 => _context.NameGenerator.GetNextTemp(typeof(ushort)),
            OpCodeTypes.Ldind_I4 => _context.NameGenerator.GetNextTemp(typeof(int)),
            OpCodeTypes.Ldind_U4 => _context.NameGenerator.GetNextTemp(typeof(uint)),
            OpCodeTypes.Ldind_I8 => _context.NameGenerator.GetNextTemp(typeof(long)),
            OpCodeTypes.Ldind_I => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldind_R4 => _context.NameGenerator.GetNextTemp(typeof(float)),
            OpCodeTypes.Ldind_R8 => _context.NameGenerator.GetNextTemp(typeof(double)),
            OpCodeTypes.Ldind_Ref => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Stind_Ref => default(VoidSsaVariable),
            OpCodeTypes.Stind_I1 => default(VoidSsaVariable),
            OpCodeTypes.Stind_I2 => default(VoidSsaVariable),
            OpCodeTypes.Stind_I4 => default(VoidSsaVariable),
            OpCodeTypes.Stind_I8 => default(VoidSsaVariable),
            OpCodeTypes.Stind_R4 => default(VoidSsaVariable),
            OpCodeTypes.Stind_R8 => default(VoidSsaVariable),
            OpCodeTypes.Add => _context.NameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Sub => _context.NameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Mul => _context.NameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Div => _context.NameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Div_Un => _context.NameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Rem => _context.NameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Rem_Un => _context.NameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.And => _context.NameGenerator.GetNextTemp(GetResultTypeFromBitManipulationOperation(operands)),
            OpCodeTypes.Or => _context.NameGenerator.GetNextTemp(GetResultTypeFromBitManipulationOperation(operands)),
            OpCodeTypes.Xor => _context.NameGenerator.GetNextTemp(GetResultTypeFromBitManipulationOperation(operands)),
            OpCodeTypes.Shl => _context.NameGenerator.GetNextTemp(GetResultTypeFromBitManipulationOperation(operands)),
            OpCodeTypes.Shr => _context.NameGenerator.GetNextTemp(GetResultTypeFromBitManipulationOperation(operands)),
            OpCodeTypes.Shr_Un => _context.NameGenerator.GetNextTemp(GetResultTypeFromBitManipulationOperation(operands)),
            OpCodeTypes.Neg => _context.NameGenerator.GetNextTemp(operands[0].Type),
            OpCodeTypes.Not => _context.NameGenerator.GetNextTemp(GetResultTypeFromBitManipulationOperation(operands)),
            OpCodeTypes.Conv_I1 => _context.NameGenerator.GetNextTemp(typeof(sbyte)),
            OpCodeTypes.Conv_I2 => _context.NameGenerator.GetNextTemp(typeof(short)),
            OpCodeTypes.Conv_I4 => _context.NameGenerator.GetNextTemp(typeof(int)),
            OpCodeTypes.Conv_I8 => _context.NameGenerator.GetNextTemp(typeof(long)),
            OpCodeTypes.Conv_R4 => _context.NameGenerator.GetNextTemp(typeof(float)),
            OpCodeTypes.Conv_R8 => _context.NameGenerator.GetNextTemp(typeof(double)),
            OpCodeTypes.Conv_U4 => _context.NameGenerator.GetNextTemp(typeof(uint)),
            OpCodeTypes.Conv_U8 => _context.NameGenerator.GetNextTemp(typeof(ulong)),
            OpCodeTypes.Cpobj => default(VoidSsaVariable),
            OpCodeTypes.Ldobj when node.Operand is MsilTypeOperand typeOperand => _context.NameGenerator.GetNextTemp(
                typeOperand
                    .Value),
            OpCodeTypes.Ldobj => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldstr => _context.NameGenerator.GetNextTemp(typeof(string)),
            OpCodeTypes.Newobj =>
                GetMethodCallReturnType(node.Operand is MsilMethodOperand operand ? operand : default),
            OpCodeTypes.Castclass => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Isinst => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Conv_R_Un => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Unbox => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Throw => default(VoidSsaVariable),
            OpCodeTypes.Ldfld => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldflda => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Stfld => default(VoidSsaVariable),
            OpCodeTypes.Ldsfld => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldsflda => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Stsfld => default(VoidSsaVariable),
            OpCodeTypes.Stobj => default(VoidSsaVariable),
            OpCodeTypes.Conv_Ovf_I1_Un => _context.NameGenerator.GetNextTemp(typeof(sbyte)),
            OpCodeTypes.Conv_Ovf_I2_Un => _context.NameGenerator.GetNextTemp(typeof(short)),
            OpCodeTypes.Conv_Ovf_I4_Un => _context.NameGenerator.GetNextTemp(typeof(int)),
            OpCodeTypes.Conv_Ovf_I8_Un => _context.NameGenerator.GetNextTemp(typeof(long)),
            OpCodeTypes.Conv_Ovf_U1_Un => _context.NameGenerator.GetNextTemp(typeof(byte)),
            OpCodeTypes.Conv_Ovf_U2_Un => _context.NameGenerator.GetNextTemp(typeof(ushort)),
            OpCodeTypes.Conv_Ovf_U4_Un => _context.NameGenerator.GetNextTemp(typeof(uint)),
            OpCodeTypes.Conv_Ovf_U8_Un => _context.NameGenerator.GetNextTemp(typeof(ulong)),
            OpCodeTypes.Conv_Ovf_I_Un when node.Operand is MsilTypeOperand type => _context.NameGenerator.GetNextTemp(
                type.Value),
            OpCodeTypes.Conv_Ovf_I_Un => throw new InvalidOperationException("The type operand is not valid."),
            OpCodeTypes.Conv_Ovf_U_Un when node.Operand is MsilTypeOperand type => _context.NameGenerator.GetNextTemp(
                type.Value),
            OpCodeTypes.Conv_Ovf_U_Un => throw new InvalidOperationException("The type operand is not valid."),
            OpCodeTypes.Box => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Newarr => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldlen => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldelema => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldelem_I1 => _context.NameGenerator.GetNextTemp(typeof(sbyte)),
            OpCodeTypes.Ldelem_U1 => _context.NameGenerator.GetNextTemp(typeof(byte)),
            OpCodeTypes.Ldelem_I2 => _context.NameGenerator.GetNextTemp(typeof(short)),
            OpCodeTypes.Ldelem_U2 => _context.NameGenerator.GetNextTemp(typeof(ushort)),
            OpCodeTypes.Ldelem_I4 => _context.NameGenerator.GetNextTemp(typeof(int)),
            OpCodeTypes.Ldelem_U4 => _context.NameGenerator.GetNextTemp(typeof(uint)),
            OpCodeTypes.Ldelem_I8 => _context.NameGenerator.GetNextTemp(typeof(long)),
            OpCodeTypes.Ldelem_I when node.Operand is MsilTypeOperand typeOperand => _context.NameGenerator.GetNextTemp(
                typeOperand.Value),
            OpCodeTypes.Ldelem_I => throw new InvalidOperationException("The type operand is not valid."),
            OpCodeTypes.Ldelem_R4 => _context.NameGenerator.GetNextTemp(typeof(float)),
            OpCodeTypes.Ldelem_R8 => _context.NameGenerator.GetNextTemp(typeof(double)),
            OpCodeTypes.Ldelem_Ref => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Stelem_I => default(VoidSsaVariable),
            OpCodeTypes.Stelem_I1 => default(VoidSsaVariable),
            OpCodeTypes.Stelem_I2 => default(VoidSsaVariable),
            OpCodeTypes.Stelem_I4 => default(VoidSsaVariable),
            OpCodeTypes.Stelem_I8 => default(VoidSsaVariable),
            OpCodeTypes.Stelem_R4 => default(VoidSsaVariable),
            OpCodeTypes.Stelem_R8 => default(VoidSsaVariable),
            OpCodeTypes.Stelem_Ref => default(VoidSsaVariable),
            OpCodeTypes.Ldelem => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Stelem => default(VoidSsaVariable),
            OpCodeTypes.Unbox_Any => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Conv_Ovf_I1 => _context.NameGenerator.GetNextTemp(typeof(sbyte)),
            OpCodeTypes.Conv_Ovf_U1 => _context.NameGenerator.GetNextTemp(typeof(byte)),
            OpCodeTypes.Conv_Ovf_I2 => _context.NameGenerator.GetNextTemp(typeof(short)),
            OpCodeTypes.Conv_Ovf_U2 => _context.NameGenerator.GetNextTemp(typeof(ushort)),
            OpCodeTypes.Conv_Ovf_I4 => _context.NameGenerator.GetNextTemp(typeof(int)),
            OpCodeTypes.Conv_Ovf_U4 => _context.NameGenerator.GetNextTemp(typeof(uint)),
            OpCodeTypes.Conv_Ovf_I8 => _context.NameGenerator.GetNextTemp(typeof(long)),
            OpCodeTypes.Conv_Ovf_U8 => _context.NameGenerator.GetNextTemp(typeof(ulong)),
            OpCodeTypes.Refanyval => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ckfinite => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Mkrefany => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldtoken => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Conv_U2 => _context.NameGenerator.GetNextTemp(typeof(ushort)),
            OpCodeTypes.Conv_U1 => _context.NameGenerator.GetNextTemp(typeof(byte)),
            OpCodeTypes.Conv_I when node.Operand is MsilTypeOperand typeOperand => _context.NameGenerator.GetNextTemp(
                typeOperand.Value),
            OpCodeTypes.Conv_I => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Conv_Ovf_I when node.Operand is MsilTypeOperand typeOperand => _context.NameGenerator.GetNextTemp(
                typeOperand.Value),
            OpCodeTypes.Conv_Ovf_I => throw new InvalidOperationException("The type operand is not valid."),
            OpCodeTypes.Conv_Ovf_U when node.Operand is MsilTypeOperand typeOperand => _context.NameGenerator.GetNextTemp(
                typeOperand.Value),
            OpCodeTypes.Conv_Ovf_U => throw new InvalidOperationException("The type operand is not valid."),
            OpCodeTypes.Add_Ovf => _context.NameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Add_Ovf_Un => _context.NameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Mul_Ovf => _context.NameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Mul_Ovf_Un => _context.NameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Sub_Ovf => _context.NameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Sub_Ovf_Un => _context.NameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Endfinally => default(VoidSsaVariable),
            OpCodeTypes.Leave => default(VoidSsaVariable),
            OpCodeTypes.Stind_I => default(VoidSsaVariable),
            OpCodeTypes.Conv_U => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Arglist => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ceq => _context.NameGenerator.GetNextTemp(typeof(bool)),
            OpCodeTypes.Cgt => _context.NameGenerator.GetNextTemp(typeof(bool)),
            OpCodeTypes.Cgt_Un => _context.NameGenerator.GetNextTemp(typeof(bool)),
            OpCodeTypes.Clt => _context.NameGenerator.GetNextTemp(typeof(bool)),
            OpCodeTypes.Clt_Un => _context.NameGenerator.GetNextTemp(typeof(bool)),
            OpCodeTypes.Ldftn => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldvirtftn => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldarg or OpCodeTypes.Ldarg_S when node.Operand is MsilInlineIntOperand operand =>
                _context.NameGenerator.GetNextTemp(_context.ArgumentSsaVariables[operand.Value].Type),
            OpCodeTypes.Ldarg or OpCodeTypes.Ldarg_S when node.Operand is MsilInlineVarOperand operand => _context.NameGenerator
                .GetNextTemp(operand.Value),
            OpCodeTypes.Ldarga => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Starg => default(VoidSsaVariable),
            OpCodeTypes.Ldloc or OpCodeTypes.Ldloc_S when node.Operand is MsilInlineIntOperand operand =>
                _context.NameGenerator.GetNextTemp(_context.LocalVariableSsaVariables[operand.Value].Type),
            OpCodeTypes.Ldloc or OpCodeTypes.Ldloc_S when node.Operand is MsilInlineVarOperand operand => _context.NameGenerator
                .GetNextTemp(operand.Value),
            OpCodeTypes.Ldloca or OpCodeTypes.Ldloca_S => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Stloc or OpCodeTypes.Stloc_S => default(VoidSsaVariable),
            OpCodeTypes.Localloc => _context.NameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Endfilter => _context.NameGenerator.GetNextTemp(typeof(nint)),
        };
    }

    private ISsaVariable GetMethodCallReturnType(MsilMethodOperand operand)
    {
        var method = operand.MethodBase;

        if (method is MethodInfo methodInfo)
        {
            return new TempSsaVariable(_context.NameGenerator.NextLocalName(), methodInfo.ReturnType);
        }

        if (method is ConstructorInfo constructorInfo)
        {
            return new TempSsaVariable(
                _context.NameGenerator.NextLocalName(),
                constructorInfo.DeclaringType ?? throw new InvalidOperationException("The constructor does not have a declaring type."));
        }

        return default(VoidSsaVariable);
    }

#pragma warning disable IDE0051, RCS1213
    private IInstruction GetInstructionV2(
        MsilInstruction<IMsilOperand> node,
        List<ISsaVariable> operands,
        ISsaVariable result,
        BasicBlock block)
#pragma warning restore IDE0051, RCS1213
    {
        _ = block;

#pragma warning disable IDE0072
        return node.IlOpCode switch
#pragma warning restore IDE0072
        {
            OpCodeTypes.Nop => default(NopInstruction),
            OpCodeTypes.Break => throw new NotSupportedException("Break is not supported."),
            OpCodeTypes.Ldarg => IrInstructionEmitter.EmitLdarg(node, result, block, _context),
            OpCodeTypes.Ldarga => throw new NotSupportedException("Ldarga is not supported."),
            OpCodeTypes.Starg => throw new NotSupportedException("Starg is not supported."),
            OpCodeTypes.Ldloc => IrInstructionEmitter.EmitLdloc(node, result, block, _context),
            OpCodeTypes.Stloc => IrInstructionEmitter.EmitStloc(node, operands, block, _context),
            OpCodeTypes.Ldnull => throw new NotSupportedException("Ldnull is not supported."),
            OpCodeTypes.Ldc_I4_M1 => throw new NotSupportedException("Ldc_I4_M1 is not supported."),
            OpCodeTypes.Ldc_I4 => throw new NotSupportedException("Ldc_I4 is not supported."),
            OpCodeTypes.Ldc_I8 => throw new NotSupportedException("Ldc_I8 is not supported."),
            OpCodeTypes.Ldc_R4 => throw new NotSupportedException("Ldc_R4 is not supported."),
            OpCodeTypes.Ldc_R8 => throw new NotSupportedException("Ldc_R8 is not supported."),
            OpCodeTypes.Dup => throw new NotSupportedException("Dup is not supported."),
            OpCodeTypes.Pop => throw new NotSupportedException("Pop is not supported."),
            OpCodeTypes.Jmp => throw new NotSupportedException("Jmp is not supported."),
            OpCodeTypes.Call or OpCodeTypes.Calli or OpCodeTypes.Callvirt => throw new NotSupportedException("Call is not supported."),
            OpCodeTypes.Ret => throw new NotSupportedException("Ret is not supported."),
            OpCodeTypes.Br => throw new NotSupportedException("Br is not supported."),
            OpCodeTypes.Brfalse => throw new NotSupportedException("Brfalse is not supported."),
            OpCodeTypes.Brtrue => throw new NotSupportedException("Brtrue is not supported."),
            OpCodeTypes.Beq => throw new NotSupportedException("Beq is not supported."),
            OpCodeTypes.Bge => throw new NotSupportedException("Bge is not supported."),
            OpCodeTypes.Bgt => throw new NotSupportedException("Bgt is not supported."),
            OpCodeTypes.Ble => throw new NotSupportedException("Ble is not supported."),
            OpCodeTypes.Blt => throw new NotSupportedException("Blt is not supported."),
            OpCodeTypes.Bne_Un => throw new NotSupportedException("Bne_Un is not supported."),
            OpCodeTypes.Bge_Un => throw new NotSupportedException("Bge_Un is not supported."),
            OpCodeTypes.Bgt_Un => throw new NotSupportedException("Bgt_Un is not supported."),
            OpCodeTypes.Ble_Un => throw new NotSupportedException("Ble_Un is not supported."),
            OpCodeTypes.Blt_Un => throw new NotSupportedException("Blt_Un is not supported."),
            OpCodeTypes.Switch => throw new NotSupportedException("Switch is not supported."),
            OpCodeTypes.Ldind_I1 => throw new NotSupportedException("Ldind_I1 is not supported."),
            OpCodeTypes.Ldind_U1 => throw new NotSupportedException("Ldind_U1 is not supported."),
            OpCodeTypes.Ldind_I2 => throw new NotSupportedException("Ldind_I2 is not supported."),
            OpCodeTypes.Ldind_U2 => throw new NotSupportedException("Ldind_U2 is not supported."),
            OpCodeTypes.Ldind_I4 => throw new NotSupportedException("Ldind_I4 is not supported."),
            OpCodeTypes.Ldind_U4 => throw new NotSupportedException("Ldind_U4 is not supported."),
            OpCodeTypes.Ldind_I8 => throw new NotSupportedException("Ldind_I8 is not supported."),
            OpCodeTypes.Ldind_I => throw new NotSupportedException("Ldind_I is not supported."),
            OpCodeTypes.Ldind_R4 => throw new NotSupportedException("Ldind_R4 is not supported."),
            OpCodeTypes.Ldind_R8 => throw new NotSupportedException("Ldind_R8 is not supported."),
            OpCodeTypes.Ldind_Ref => throw new NotSupportedException("Ldind_Ref is not supported."),
            OpCodeTypes.Stind_Ref => throw new NotSupportedException("Stind_Ref is not supported."),
            OpCodeTypes.Stind_I1 => throw new NotSupportedException("Stind_I1 is not supported."),
            OpCodeTypes.Stind_I2 => throw new NotSupportedException("Stind_I2 is not supported."),
            OpCodeTypes.Stind_I4 => throw new NotSupportedException("Stind_I4 is not supported."),
            OpCodeTypes.Stind_I8 => throw new NotSupportedException("Stind_I8 is not supported."),
            OpCodeTypes.Stind_R4 => throw new NotSupportedException("Stind_R4 is not supported."),
            OpCodeTypes.Stind_R8 => throw new NotSupportedException("Stind_R8 is not supported."),
            OpCodeTypes.Add => throw new NotSupportedException("Add is not supported."),
            OpCodeTypes.Sub => throw new NotSupportedException("Sub is not supported."),
            OpCodeTypes.Mul => throw new NotSupportedException("Mul is not supported."),
            OpCodeTypes.Div => throw new NotSupportedException("Div is not supported."),
            OpCodeTypes.Div_Un => throw new NotSupportedException("Div_Un is not supported."),
            OpCodeTypes.Rem => throw new NotSupportedException("Rem is not supported."),
            OpCodeTypes.Rem_Un => throw new NotSupportedException("Rem_Un is not supported."),
            OpCodeTypes.And => throw new NotSupportedException("And is not supported."),
            OpCodeTypes.Or => throw new NotSupportedException("Or is not supported."),
            OpCodeTypes.Xor => throw new NotSupportedException("Xor is not supported."),
            OpCodeTypes.Shl => throw new NotSupportedException("Shl is not supported."),
            OpCodeTypes.Shr => throw new NotSupportedException("Shr is not supported."),
            OpCodeTypes.Shr_Un => throw new NotSupportedException("Shr_Un is not supported."),
            OpCodeTypes.Neg => throw new NotSupportedException("Neg is not supported."),
            OpCodeTypes.Not => throw new NotSupportedException("Not is not supported."),
            OpCodeTypes.Conv_I1 => throw new NotSupportedException("Conv_I1 is not supported."),
            OpCodeTypes.Conv_I2 => throw new NotSupportedException("Conv_I2 is not supported."),
            OpCodeTypes.Conv_I4 => throw new NotSupportedException("Conv_I4 is not supported."),
            OpCodeTypes.Conv_I8 => throw new NotSupportedException("Conv_I8 is not supported."),
            OpCodeTypes.Conv_R4 => throw new NotSupportedException("Conv_R4 is not supported."),
            OpCodeTypes.Conv_R8 => throw new NotSupportedException("Conv_R8 is not supported."),
            OpCodeTypes.Conv_U4 => throw new NotSupportedException("Conv_U4 is not supported."),
            OpCodeTypes.Conv_U8 => throw new NotSupportedException("Conv_U8 is not supported."),
            OpCodeTypes.Cpobj => throw new NotSupportedException("Cpobj is not supported."),
            OpCodeTypes.Ldobj => throw new NotSupportedException("Ldobj is not supported."),
            OpCodeTypes.Ldstr => throw new NotSupportedException("Ldstr is not supported."),
            OpCodeTypes.Newobj => throw new NotSupportedException("Newobj is not supported."),
            OpCodeTypes.Castclass => throw new NotSupportedException("Castclass is not supported."),
            OpCodeTypes.Isinst => throw new NotSupportedException("Isinst is not supported."),
            OpCodeTypes.Conv_R_Un => throw new NotSupportedException("Conv_R_Un is not supported."),
            OpCodeTypes.Unbox => throw new NotSupportedException("Unbox is not supported."),
            OpCodeTypes.Throw => throw new NotSupportedException("Throw is not supported."),
            OpCodeTypes.Ldfld => throw new NotSupportedException("Ldfld is not supported."),
            OpCodeTypes.Ldflda => throw new NotSupportedException("Ldflda is not supported."),
            OpCodeTypes.Stfld => throw new NotSupportedException("Stfld is not supported."),
            OpCodeTypes.Ldsfld => throw new NotSupportedException("Ldsfld is not supported."),
            OpCodeTypes.Ldsflda => throw new NotSupportedException("Ldsflda is not supported."),
            OpCodeTypes.Stsfld => throw new NotSupportedException("Stsfld is not supported."),
            OpCodeTypes.Stobj => throw new NotSupportedException("Stobj is not supported."),
            OpCodeTypes.Conv_Ovf_I1_Un => throw new NotSupportedException("Conv_Ovf_I1_Un is not supported."),
            OpCodeTypes.Conv_Ovf_I2_Un => throw new NotSupportedException("Conv_Ovf_I2_Un is not supported."),
            OpCodeTypes.Conv_Ovf_I4_Un => throw new NotSupportedException("Conv_Ovf_I4_Un is not supported."),
            OpCodeTypes.Conv_Ovf_I8_Un => throw new NotSupportedException("Conv_Ovf_I8_Un is not supported."),
            OpCodeTypes.Conv_Ovf_U1_Un => throw new NotSupportedException("Conv_Ovf_U1_Un is not supported."),
            OpCodeTypes.Conv_Ovf_U2_Un => throw new NotSupportedException("Conv_Ovf_U2_Un is not supported."),
            OpCodeTypes.Conv_Ovf_U4_Un => throw new NotSupportedException("Conv_Ovf_U4_Un is not supported."),
            OpCodeTypes.Conv_Ovf_U8_Un => throw new NotSupportedException("Conv_Ovf_U8_Un is not supported."),
            OpCodeTypes.Conv_Ovf_I_Un => throw new NotSupportedException("Conv_Ovf_I_Un is not supported."),
            OpCodeTypes.Conv_Ovf_U_Un => throw new NotSupportedException("Conv_Ovf_U_Un is not supported."),
            OpCodeTypes.Box => throw new NotSupportedException("Box is not supported."),
            OpCodeTypes.Newarr => throw new NotSupportedException("Newarr is not supported."),
            OpCodeTypes.Ldlen => throw new NotSupportedException("Ldlen is not supported."),
            OpCodeTypes.Ldelema => throw new NotSupportedException("Ldelema is not supported."),
            OpCodeTypes.Ldelem_I1 => throw new NotSupportedException("Ldelem_I1 is not supported."),
            OpCodeTypes.Ldelem_U1 => throw new NotSupportedException("Ldelem_U1 is not supported."),
            OpCodeTypes.Ldelem_I2 => throw new NotSupportedException("Ldelem_I2 is not supported."),
            OpCodeTypes.Ldelem_U2 => throw new NotSupportedException("Ldelem_U2 is not supported."),
            OpCodeTypes.Ldelem_I4 => throw new NotSupportedException("Ldelem_I4 is not supported."),
            OpCodeTypes.Ldelem_U4 => throw new NotSupportedException("Ldelem_U4 is not supported."),
            OpCodeTypes.Ldelem_I8 => throw new NotSupportedException("Ldelem_I8 is not supported."),
            OpCodeTypes.Ldelem_I => throw new NotSupportedException("Ldelem_I is not supported."),
            OpCodeTypes.Ldelem_R4 => throw new NotSupportedException("Ldelem_R4 is not supported."),
            OpCodeTypes.Ldelem_R8 => throw new NotSupportedException("Ldelem_R8 is not supported."),
            OpCodeTypes.Ldelem_Ref => throw new NotSupportedException("Ldelem_Ref is not supported."),
            OpCodeTypes.Stelem_I => throw new NotSupportedException("Stelem_I is not supported."),
            OpCodeTypes.Stelem_I1 => throw new NotSupportedException("Stelem_I1 is not supported."),
            OpCodeTypes.Stelem_I2 => throw new NotSupportedException("Stelem_I2 is not supported."),
            OpCodeTypes.Stelem_I4 => throw new NotSupportedException("Stelem_I4 is not supported."),
            OpCodeTypes.Stelem_I8 => throw new NotSupportedException("Stelem_I8 is not supported."),
            OpCodeTypes.Stelem_R4 => throw new NotSupportedException("Stelem_R4 is not supported."),
            OpCodeTypes.Stelem_R8 => throw new NotSupportedException("Stelem_R8 is not supported."),
            OpCodeTypes.Stelem_Ref => throw new NotSupportedException("Stelem_Ref is not supported."),
            OpCodeTypes.Ldelem => throw new NotSupportedException("Ldelem is not supported."),
            OpCodeTypes.Stelem => throw new NotSupportedException("Stelem is not supported."),
            OpCodeTypes.Unbox_Any => throw new NotSupportedException("Unbox_Any is not supported."),
            OpCodeTypes.Conv_Ovf_I1 => throw new NotSupportedException("Conv_Ovf_I1 is not supported."),
            OpCodeTypes.Conv_Ovf_U1 => throw new NotSupportedException("Conv_Ovf_U1 is not supported."),
            OpCodeTypes.Conv_Ovf_I2 => throw new NotSupportedException("Conv_Ovf_I2 is not supported."),
            OpCodeTypes.Conv_Ovf_U2 => throw new NotSupportedException("Conv_Ovf_U2 is not supported."),
            OpCodeTypes.Conv_Ovf_I4 => throw new NotSupportedException("Conv_Ovf_I4 is not supported."),
            OpCodeTypes.Conv_Ovf_U4 => throw new NotSupportedException("Conv_Ovf_U4 is not supported."),
            OpCodeTypes.Conv_Ovf_I8 => throw new NotSupportedException("Conv_Ovf_I8 is not supported."),
            OpCodeTypes.Conv_Ovf_U8 => throw new NotSupportedException("Conv_Ovf_U8 is not supported."),
            OpCodeTypes.Refanyval => throw new NotSupportedException("Refanyval is not supported."),
            OpCodeTypes.Ckfinite => throw new NotSupportedException("Ckfinite is not supported."),
            OpCodeTypes.Mkrefany => throw new NotSupportedException("Mkrefany is not supported."),
            OpCodeTypes.Ldtoken => throw new NotSupportedException("Ldtoken is not supported."),
            OpCodeTypes.Conv_U2 => throw new NotSupportedException("Conv_U2 is not supported."),
            OpCodeTypes.Conv_U1 => throw new NotSupportedException("Conv_U1 is not supported."),
            OpCodeTypes.Conv_I => throw new NotSupportedException("Conv_I is not supported."),
            OpCodeTypes.Conv_Ovf_I => throw new NotSupportedException("Conv_Ovf_I is not supported."),
            OpCodeTypes.Conv_Ovf_U => throw new NotSupportedException("Conv_Ovf_U is not supported."),
            OpCodeTypes.Add_Ovf => throw new NotSupportedException("Add_Ovf is not supported."),
            OpCodeTypes.Add_Ovf_Un => throw new NotSupportedException("Add_Ovf_Un is not supported."),
            OpCodeTypes.Mul_Ovf => throw new NotSupportedException("Mul_Ovf is not supported."),
            OpCodeTypes.Mul_Ovf_Un => throw new NotSupportedException("Mul_Ovf_Un is not supported."),
            OpCodeTypes.Sub_Ovf => throw new NotSupportedException("Sub_Ovf is not supported."),
            OpCodeTypes.Sub_Ovf_Un => throw new NotSupportedException("Sub_Ovf_Un is not supported."),
            OpCodeTypes.Endfinally => throw new NotSupportedException("Endfinally is not supported."),
            OpCodeTypes.Leave => throw new NotSupportedException("Leave is not supported."),
            OpCodeTypes.Stind_I => throw new NotSupportedException("Stind_I is not supported."),
            OpCodeTypes.Conv_U => throw new NotSupportedException("Conv_U is not supported."),
            OpCodeTypes.Arglist => throw new NotSupportedException("Arglist is not supported."),
            OpCodeTypes.Ceq => throw new NotSupportedException("Ceq is not supported."),
            OpCodeTypes.Cgt => throw new NotSupportedException("Cgt is not supported."),
            OpCodeTypes.Cgt_Un => throw new NotSupportedException("Cgt_Un is not supported."),
            OpCodeTypes.Clt => throw new NotSupportedException("Clt is not supported."),
            OpCodeTypes.Clt_Un => throw new NotSupportedException("Clt_Un is not supported."),
            OpCodeTypes.Ldftn => throw new NotSupportedException("Ldftn is not supported."),
            OpCodeTypes.Ldvirtftn => throw new NotSupportedException("Ldvirtftn is not supported."),
            OpCodeTypes.Localloc => throw new NotSupportedException("Localloc is not supported."),
            OpCodeTypes.Endfilter => throw new NotSupportedException("Endfilter is not supported."),
            OpCodeTypes.Unaligned => throw new NotSupportedException("Unaligned is not supported."),
            OpCodeTypes.Volatile => throw new NotSupportedException("Volatile is not supported."),
            OpCodeTypes.Tailcall => throw new NotSupportedException("Tailcall is not supported."),
            OpCodeTypes.Initobj => throw new NotSupportedException("Initobj is not supported."),
            OpCodeTypes.Constrained => throw new NotSupportedException("Constrained is not supported."),
            OpCodeTypes.Cpblk => throw new NotSupportedException("Cpblk is not supported."),
            OpCodeTypes.Initblk => throw new NotSupportedException("Initblk is not supported."),
            OpCodeTypes.Rethrow => throw new NotSupportedException("Rethrow is not supported."),
            OpCodeTypes.Sizeof => throw new NotSupportedException("Sizeof is not supported."),
            OpCodeTypes.Refanytype => throw new NotSupportedException("Refanytype is not supported."),
            OpCodeTypes.Readonly => throw new NotSupportedException("Readonly is not supported."),
            _ => throw new NotSupportedException("The instruction is not supported.")
        };
    }

    private IInstruction GetInstruction(
        MsilInstruction<IMsilOperand> node,
        List<ISsaVariable> operands,
        ISsaVariable result,
        BasicBlock block,
        BasicBlock nextBlock,
        BasicBlock[] basicBlocks)
    {
        _ = block;

#pragma warning disable IDE0072
        return node.IlOpCode switch
#pragma warning restore IDE0072
        {
            OpCodeTypes.Nop => default(NopInstruction),
            OpCodeTypes.Ldarg or OpCodeTypes.Ldarg_S when node.Operand is MsilInlineIntOperand operand => new
                LoadArgumentInstruction
                {
                    Parameter = _context.ArgumentSsaVariables[operand.Value].ToIrValue(),
                    Result = result.ToIrValue(),
                    MsilInstruction = node,
                    Block = block
                },
            OpCodeTypes.Ldarg or OpCodeTypes.Ldarg_S when node.Operand is MsilInlineVarOperand operand => new
                LoadArgumentInstruction
                {
                    Parameter = _context.ArgumentSsaVariables[operand.Index].ToIrValue(),
                    Result = result.ToIrValue(),
                    MsilInstruction = node,
                    Block = block
                },
            OpCodeTypes.Ldloc or OpCodeTypes.Ldloc_S when node.Operand is MsilInlineIntOperand operand => new
                LoadLocalInstruction
                {
                    Local = _context.LocalVariableSsaVariables[operand.Value].ToIrValue(),
                    Result = result.ToIrValue(),
                    MsilInstruction = node,
                    Block = block
                },
            OpCodeTypes.Ldloc or OpCodeTypes.Ldloc_S when node.Operand is MsilInlineVarOperand operand => new
                LoadLocalInstruction
                {
                    Local = _context.LocalVariableSsaVariables[operand.Index].ToIrValue(),
                    Result = result.ToIrValue(),
                    MsilInstruction = node,
                    Block = block
                },
            OpCodeTypes.Ldc_I4 or OpCodeTypes.Ldc_I4_S when node.Operand is MsilInlineIntOperand operand => new
                LoadConstantInstruction
                {
                    Result = result.ToIrValue(),
                    Value = operand.Value,
                    MsilInstruction = node,
                    Block = block
                },
            OpCodeTypes.Ldc_I8 when node.Operand is MsilInlineLongOperand operand => new LoadConstantInstruction
            {
                Result = result.ToIrValue(),
                Value = operand.Value,
                MsilInstruction = node,
                Block = block
            },
            OpCodeTypes.Ldc_R4 when node.Operand is MsilInlineSingleOperand operand => new LoadConstantInstruction
            {
                Result = result.ToIrValue(),
                Value = operand.Value,
                MsilInstruction = node,
                Block = block
            },
            OpCodeTypes.Ldc_R8 when node.Operand is MsilInlineDoubleOperand operand => new LoadConstantInstruction
            {
                Result = result.ToIrValue(),
                Value = operand.Value,
                MsilInstruction = node,
                Block = block
            },
            OpCodeTypes.Conv_I or OpCodeTypes.Conv_I1 or OpCodeTypes.Conv_I2 or OpCodeTypes.Conv_I4
                or OpCodeTypes.Conv_I8 or OpCodeTypes.Conv_R4 or OpCodeTypes.Conv_R8 or OpCodeTypes.Conv_U4
                or OpCodeTypes.Conv_U8 => new SignExtendInstruction
                {
                    Result = result.ToIrValue(),
                    Value = operands[0].ToIrValue(),
                    MsilInstruction = node,
                    Block = block
                },
            OpCodeTypes.Br or OpCodeTypes.Br_S when node.Operand is MsilBranchTargetOperand operand => new
                BranchInstruction
                {
                    Target = FindBlockForOffset(basicBlocks, operand.Target),
                    MsilInstruction = node,
                    Block = block
                },
            OpCodeTypes.Brtrue or OpCodeTypes.Brtrue_S when node.Operand is MsilBranchTargetOperand operand => new
                ConditionalBranchInstruction
                {
                    Condition = operands[0].ToIrValue(),
                    Target = FindBlockForOffset(basicBlocks, operand.Target),
                    FalseTarget = nextBlock,
                    MsilInstruction = node,
                    Block = block
                },
            OpCodeTypes.Brfalse or OpCodeTypes.Brfalse_S when node.Operand is MsilBranchTargetOperand operand => new
                ConditionalBranchInstruction
                {
                    Condition = operands[0].ToIrValue(),
                    Target = nextBlock,
                    FalseTarget = FindBlockForOffset(basicBlocks, operand.Target),
                    MsilInstruction = node,
                    Block = block
                },
            OpCodeTypes.Bge or OpCodeTypes.Bge_S when node.Operand is MsilBranchTargetOperand operand => new
                BranchGreaterThanInstruction
                {
                    Left = operands[0].ToIrValue(),
                    Right = operands[1].ToIrValue(),
                    Target = FindBlockForOffset(basicBlocks, operand.Target),
                    FalseTarget = nextBlock,
                    MsilInstruction = node,
                    Block = block
                },
            OpCodeTypes.Mul => new MultiplyInstruction
            {
                Result = result.ToIrValue(),
                Left = operands[0].ToIrValue(),
                Right = operands[1].ToIrValue(),
                MsilInstruction = node,
                Block = block
            },
            OpCodeTypes.Add => new AddInstruction
            {
                Result = result.ToIrValue(),
                Left = operands[0].ToIrValue(),
                Right = operands[1].ToIrValue(),
                MsilInstruction = node,
                Block = block
            },
            OpCodeTypes.Ldind_I or OpCodeTypes.Ldind_I1 or OpCodeTypes.Ldind_I2 or OpCodeTypes.Ldind_I4
                or OpCodeTypes.Ldind_I8 or OpCodeTypes.Ldind_U1 or OpCodeTypes.Ldind_U2 or OpCodeTypes.Ldind_U4
                or OpCodeTypes.Ldind_R4 or OpCodeTypes.Ldind_R8 =>
                new LoadElementFromPointerInstruction
                {
                    Result = result.ToIrValue(),
                    Pointer = operands[0].ToIrValue(),
                    MsilInstruction = node,
                    Block = block
                },
            OpCodeTypes.Stind_I or OpCodeTypes.Stind_I1 or OpCodeTypes.Stind_I2 or OpCodeTypes.Stind_I4
                or OpCodeTypes.Stind_I8 or OpCodeTypes.Stind_R4 or OpCodeTypes.Stind_R8 =>
                new StoreElementAtPointerInstruction
                {
                    Value = operands[0].ToIrValue(),
                    Pointer = operands[1].ToIrValue(),
                    MsilInstruction = node,
                    Block = block
                },
            OpCodeTypes.Clt => new CompareLessThanInstruction
            {
                Result = result.ToIrValue(),
                Left = operands[0].ToIrValue(),
                Right = operands[1].ToIrValue(),
                MsilInstruction = node,
                Block = block
            },
            OpCodeTypes.Ret when operands.Count == 0 || _context.DisassembledMethod.Metadata.ReturnType == typeof(void) => new ReturnVoidInstruction
            {
                MsilInstruction = node,
                Block = block
            },
            OpCodeTypes.Ret when operands.Count != 0 => new ReturnInstruction
            {
                Value = operands[0].ToIrValue(),
                MsilInstruction = node,
                Block = block
            },
            OpCodeTypes.Call or OpCodeTypes.Calli or OpCodeTypes.Callvirt when node.Operand is MsilMethodOperand operand =>
                new CallInstruction
                {
                    MethodBase = operand.MethodBase ?? throw new InvalidOperationException("The method operand is null."),
                    Result = result.ToIrValue(),
                    Arguments = operands.Select(o => o.ToIrValue()).ToImmutableArray(),
                    MsilInstruction = node,
                    Block = block
                },
            OpCodeTypes.Ldloca or OpCodeTypes.Ldloca_S when node.Operand is MsilInlineIntOperand operand => new LoadLocalAddressInstruction
            {
                Result = result.ToIrValue(),
                Local = _context.LocalVariableSsaVariables[operand.Value].ToIrValue(),
                MsilInstruction = node,
                Block = block
            },
            OpCodeTypes.Ldloca or OpCodeTypes.Ldloca_S when node.Operand is MsilInlineVarOperand operand => new LoadLocalAddressInstruction
            {
                Result = result.ToIrValue(),
                Local = _context.LocalVariableSsaVariables[operand.Index].ToIrValue(),
                MsilInstruction = node,
                Block = block
            },
        };
    }
}