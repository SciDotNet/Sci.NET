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

    private readonly DisassembledMsilMethod _disassembledMethod;
    private readonly LocalVariableSsaVariable[] _localVariableSsaVariables;
    private readonly ParameterSsaVariable[] _argumentSsaVariables;
    private readonly VariableNameGenerator _nameGenerator;
    private readonly Stack<ISsaVariable> _stack;

    /// <summary>
    /// Initializes a new instance of the <see cref="MsilToIrTranslator"/> class.
    /// </summary>
    /// <param name="disassembledMethod">The disassembled method.</param>
    public MsilToIrTranslator(DisassembledMsilMethod disassembledMethod)
    {
        _disassembledMethod = disassembledMethod;
        _nameGenerator = new VariableNameGenerator();
        _localVariableSsaVariables = new LocalVariableSsaVariable[disassembledMethod.Metadata.Variables.Length];
        _stack = new Stack<ISsaVariable>();

        for (var index = 0; index < disassembledMethod.Metadata.Variables.Length; index++)
        {
            _localVariableSsaVariables[index] = new LocalVariableSsaVariable(
                index,
                _nameGenerator.NextLocalName(),
                disassembledMethod.Metadata.Variables[index].Type);
        }

        _argumentSsaVariables = new ParameterSsaVariable[disassembledMethod.Metadata.Parameters.Length];

        for (var index = 0; index < disassembledMethod.Metadata.Parameters.Length; index++)
        {
            _argumentSsaVariables[index] = new ParameterSsaVariable(
                index,
                $"arg_{disassembledMethod.Metadata.Parameters[index].Name}",
                disassembledMethod.Metadata.Parameters[index].ParameterType);
        }
    }

    /// <summary>
    /// Transforms the method into SSA form.
    /// </summary>
    /// <returns>The SSA form of the method.</returns>
    public MsilSsaMethod Transform()
    {
        var basicBlocks = FindBasicBlocks();

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
                    poppedItems[count - 1] = _stack.Pop();
                    count--;
                }

                for (var i = 0; i < popCount; i++)
                {
                    operands.Add(poppedItems[i]);
                }

                var result = GetResult(node, operands);

                if (result.Type != typeof(void))
                {
                    _stack.Push(result);
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
                blockInstructions.Add(new BranchInstruction
                {
                    Target = basicBlocks[blockIdx + 1], MsilInstruction = null
                });
            }

            block.SetInstructions(blockInstructions);
        }

        foreach (var basicBlock in basicBlocks)
        {
            foreach (var transform in TransformManager.GetTransforms())
            {
                transform.Transform(basicBlock);
            }
        }

        return new MsilSsaMethod
        {
            BasicBlocks = basicBlocks,
            Locals = _localVariableSsaVariables,
            Parameters = _argumentSsaVariables,
            ReturnType = _disassembledMethod.Metadata.ReturnType,
            Metadata = _disassembledMethod.Metadata
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

    private BasicBlock[] FindBasicBlocks()
    {
        var leaders = new List<int> { 0 };

        foreach (var instruction in _disassembledMethod.Instructions)
        {
            if (instruction.IsBranch)
            {
                leaders.AddRange(instruction.GetBranchTargets());

                if (instruction.IsConditionalBranch)
                {
                    leaders.Add(instruction.Offset);
                }
            }
        }

        leaders.Sort();

        var basicBlocks = new List<BasicBlock>();
        var instructions = _disassembledMethod.Instructions.ToArray();
        var offsetToIndex = new Dictionary<int, int>();

        for (var i = 0; i < instructions.Length; i++)
        {
            offsetToIndex.Add(instructions[i].Offset, i);
        }

        var firstBlockInstructions = instructions[..offsetToIndex[leaders[1]]];

        basicBlocks.Add(new BasicBlock("block_1", firstBlockInstructions));

        for (var i = 1; i < leaders.Count - 1; i++)
        {
            var blockInstructions = instructions[offsetToIndex[leaders[i]].. (offsetToIndex[leaders[i + 1]] + 1)];
            basicBlocks.Add(new BasicBlock($"block_{i + 1}", blockInstructions));
        }

        if (leaders[^1] != instructions[^1].Offset)
        {
            // The last block is not a leader.
            var lastBlockInstructions = instructions[offsetToIndex[leaders[^1]]..];
            basicBlocks.Add(new BasicBlock($"block_{leaders.Count}", lastBlockInstructions));
        }

        var firstBlock = GetLocalsForBlock(basicBlocks[0]);
        basicBlocks.Insert(0, firstBlock);

        return basicBlocks.ToArray();
    }

    private BasicBlock GetLocalsForBlock(BasicBlock basicBlock)
    {
        var locals = new List<IInstruction>();

        foreach (var local in _localVariableSsaVariables)
        {
            var irLocal = local.ToIrValue();
            locals.Add(new DeclareLocalInstruction
            {
                Result = local.ToIrValue(), Value = irLocal.Type.CreateDefaultInstance(), MsilInstruction = null
            });
        }

        locals.Add(new BranchInstruction { Target = basicBlock, MsilInstruction = null });

        return new BasicBlock("block_0", locals);
    }

    private List<ISsaVariable> ExtractOperands(MsilInstruction<IMsilOperand> node)
    {
        var operands = new List<ISsaVariable>();
#pragma warning disable IDE0010
        switch (node.IlOpCode)
#pragma warning restore IDE0010
        {
            case OpCodeTypes.Ldarg or OpCodeTypes.Ldarg_S when node.Operand is MsilInlineIntOperand index:
                operands.Add(_argumentSsaVariables[index.Value]);
                break;
            case OpCodeTypes.Ldloc or OpCodeTypes.Ldloc_S when node.Operand is MsilInlineIntOperand index:
                operands.Add(_localVariableSsaVariables[index.Value]);
                break;
            case OpCodeTypes.Ldc_I4 or OpCodeTypes.Ldc_I4_S when node.Operand is MsilInlineIntOperand value:
                operands.Add(new Int4ConstantSsaVariable(_nameGenerator.NextTemporaryName(), value.Value));
                break;
            case OpCodeTypes.Ldc_I8 when node.Operand is MsilInlineLongOperand value:
                operands.Add(new Int8ConstantSsaVariable(_nameGenerator.NextTemporaryName(), value.Value));
                break;
            case OpCodeTypes.Ldc_R4 when node.Operand is MsilInlineSingleOperand value:
                operands.Add(new Float4ConstantSsaVariable(_nameGenerator.NextTemporaryName(), value.Value));
                break;
            case OpCodeTypes.Ldc_R8 when node.Operand is MsilInlineDoubleOperand value:
                operands.Add(new Float8ConstantSsaVariable(_nameGenerator.NextTemporaryName(), value.Value));
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
            PopBehaviour.Varpop when node.IlOpCode is OpCodeTypes.Ret && _stack.Count == 1 => 1,
            PopBehaviour.Varpop when node.IlOpCode is OpCodeTypes.Ret && _stack.Count == 0 &&
                                     _disassembledMethod.Metadata.ReturnType == typeof(void) => 0,
            PopBehaviour.Varpop when node.IlOpCode is OpCodeTypes.Ret && _stack.Count > 0 =>
                throw new InvalidOperationException("The stack is not empty."),
            PopBehaviour.Varpop when node.Operand is MsilMethodOperand operand => GetMethodCallPopBehaviour(operand),
            PopBehaviour.Varpop => throw new NotSupportedException("Varpop is not supported."),
            _ => throw new InvalidOperationException("The pop behaviour is not supported.")
        };
    }

    private ISsaVariable GetResult(MsilInstruction<IMsilOperand> node, List<ISsaVariable> operands)
    {
#pragma warning disable IDE0072
        return node.IlOpCode switch
#pragma warning restore IDE0072
        {
            OpCodeTypes.Nop => default(VoidSsaVariable),
            OpCodeTypes.Break => default(VoidSsaVariable),
            OpCodeTypes.Ldnull => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldc_I4 or OpCodeTypes.Ldc_I4_S => _nameGenerator.GetNextTemp(typeof(int)),
            OpCodeTypes.Ldc_I8 => _nameGenerator.GetNextTemp(typeof(long)),
            OpCodeTypes.Ldc_R4 => _nameGenerator.GetNextTemp(typeof(float)),
            OpCodeTypes.Ldc_R8 => _nameGenerator.GetNextTemp(typeof(double)),
            OpCodeTypes.Dup => operands[0],
            OpCodeTypes.Pop => _stack.Peek(),
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
            OpCodeTypes.Ldind_I1 => _nameGenerator.GetNextTemp(typeof(sbyte)),
            OpCodeTypes.Ldind_U1 => _nameGenerator.GetNextTemp(typeof(byte)),
            OpCodeTypes.Ldind_I2 => _nameGenerator.GetNextTemp(typeof(short)),
            OpCodeTypes.Ldind_U2 => _nameGenerator.GetNextTemp(typeof(ushort)),
            OpCodeTypes.Ldind_I4 => _nameGenerator.GetNextTemp(typeof(int)),
            OpCodeTypes.Ldind_U4 => _nameGenerator.GetNextTemp(typeof(uint)),
            OpCodeTypes.Ldind_I8 => _nameGenerator.GetNextTemp(typeof(long)),
            OpCodeTypes.Ldind_I => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldind_R4 => _nameGenerator.GetNextTemp(typeof(float)),
            OpCodeTypes.Ldind_R8 => _nameGenerator.GetNextTemp(typeof(double)),
            OpCodeTypes.Ldind_Ref => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Stind_Ref => default(VoidSsaVariable),
            OpCodeTypes.Stind_I1 => default(VoidSsaVariable),
            OpCodeTypes.Stind_I2 => default(VoidSsaVariable),
            OpCodeTypes.Stind_I4 => default(VoidSsaVariable),
            OpCodeTypes.Stind_I8 => default(VoidSsaVariable),
            OpCodeTypes.Stind_R4 => default(VoidSsaVariable),
            OpCodeTypes.Stind_R8 => default(VoidSsaVariable),
            OpCodeTypes.Add => _nameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Sub => _nameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Mul => _nameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Div => _nameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Div_Un => _nameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Rem => _nameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Rem_Un => _nameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.And => _nameGenerator.GetNextTemp(GetResultTypeFromBitManipulationOperation(operands)),
            OpCodeTypes.Or => _nameGenerator.GetNextTemp(GetResultTypeFromBitManipulationOperation(operands)),
            OpCodeTypes.Xor => _nameGenerator.GetNextTemp(GetResultTypeFromBitManipulationOperation(operands)),
            OpCodeTypes.Shl => _nameGenerator.GetNextTemp(GetResultTypeFromBitManipulationOperation(operands)),
            OpCodeTypes.Shr => _nameGenerator.GetNextTemp(GetResultTypeFromBitManipulationOperation(operands)),
            OpCodeTypes.Shr_Un => _nameGenerator.GetNextTemp(GetResultTypeFromBitManipulationOperation(operands)),
            OpCodeTypes.Neg => _nameGenerator.GetNextTemp(operands[0].Type),
            OpCodeTypes.Not => _nameGenerator.GetNextTemp(GetResultTypeFromBitManipulationOperation(operands)),
            OpCodeTypes.Conv_I1 => _nameGenerator.GetNextTemp(typeof(sbyte)),
            OpCodeTypes.Conv_I2 => _nameGenerator.GetNextTemp(typeof(short)),
            OpCodeTypes.Conv_I4 => _nameGenerator.GetNextTemp(typeof(int)),
            OpCodeTypes.Conv_I8 => _nameGenerator.GetNextTemp(typeof(long)),
            OpCodeTypes.Conv_R4 => _nameGenerator.GetNextTemp(typeof(float)),
            OpCodeTypes.Conv_R8 => _nameGenerator.GetNextTemp(typeof(double)),
            OpCodeTypes.Conv_U4 => _nameGenerator.GetNextTemp(typeof(uint)),
            OpCodeTypes.Conv_U8 => _nameGenerator.GetNextTemp(typeof(ulong)),
            OpCodeTypes.Cpobj => default(VoidSsaVariable),
            OpCodeTypes.Ldobj when node.Operand is MsilTypeOperand typeOperand => _nameGenerator.GetNextTemp(typeOperand
                .Value),
            OpCodeTypes.Ldobj => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldstr => _nameGenerator.GetNextTemp(typeof(string)),
            OpCodeTypes.Newobj =>
                GetMethodCallReturnType(node.Operand is MsilMethodOperand operand ? operand : default),
            OpCodeTypes.Castclass => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Isinst => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Conv_R_Un => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Unbox => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Throw => default(VoidSsaVariable),
            OpCodeTypes.Ldfld => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldflda => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Stfld => default(VoidSsaVariable),
            OpCodeTypes.Ldsfld => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldsflda => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Stsfld => default(VoidSsaVariable),
            OpCodeTypes.Stobj => default(VoidSsaVariable),
            OpCodeTypes.Conv_Ovf_I1_Un => _nameGenerator.GetNextTemp(typeof(sbyte)),
            OpCodeTypes.Conv_Ovf_I2_Un => _nameGenerator.GetNextTemp(typeof(short)),
            OpCodeTypes.Conv_Ovf_I4_Un => _nameGenerator.GetNextTemp(typeof(int)),
            OpCodeTypes.Conv_Ovf_I8_Un => _nameGenerator.GetNextTemp(typeof(long)),
            OpCodeTypes.Conv_Ovf_U1_Un => _nameGenerator.GetNextTemp(typeof(byte)),
            OpCodeTypes.Conv_Ovf_U2_Un => _nameGenerator.GetNextTemp(typeof(ushort)),
            OpCodeTypes.Conv_Ovf_U4_Un => _nameGenerator.GetNextTemp(typeof(uint)),
            OpCodeTypes.Conv_Ovf_U8_Un => _nameGenerator.GetNextTemp(typeof(ulong)),
            OpCodeTypes.Conv_Ovf_I_Un when node.Operand is MsilTypeOperand type => _nameGenerator.GetNextTemp(
                type.Value),
            OpCodeTypes.Conv_Ovf_I_Un => throw new InvalidOperationException("The type operand is not valid."),
            OpCodeTypes.Conv_Ovf_U_Un when node.Operand is MsilTypeOperand type => _nameGenerator.GetNextTemp(
                type.Value),
            OpCodeTypes.Conv_Ovf_U_Un => throw new InvalidOperationException("The type operand is not valid."),
            OpCodeTypes.Box => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Newarr => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldlen => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldelema => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldelem_I1 => _nameGenerator.GetNextTemp(typeof(sbyte)),
            OpCodeTypes.Ldelem_U1 => _nameGenerator.GetNextTemp(typeof(byte)),
            OpCodeTypes.Ldelem_I2 => _nameGenerator.GetNextTemp(typeof(short)),
            OpCodeTypes.Ldelem_U2 => _nameGenerator.GetNextTemp(typeof(ushort)),
            OpCodeTypes.Ldelem_I4 => _nameGenerator.GetNextTemp(typeof(int)),
            OpCodeTypes.Ldelem_U4 => _nameGenerator.GetNextTemp(typeof(uint)),
            OpCodeTypes.Ldelem_I8 => _nameGenerator.GetNextTemp(typeof(long)),
            OpCodeTypes.Ldelem_I when node.Operand is MsilTypeOperand typeOperand => _nameGenerator.GetNextTemp(
                typeOperand.Value),
            OpCodeTypes.Ldelem_I => throw new InvalidOperationException("The type operand is not valid."),
            OpCodeTypes.Ldelem_R4 => _nameGenerator.GetNextTemp(typeof(float)),
            OpCodeTypes.Ldelem_R8 => _nameGenerator.GetNextTemp(typeof(double)),
            OpCodeTypes.Ldelem_Ref => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Stelem_I => default(VoidSsaVariable),
            OpCodeTypes.Stelem_I1 => default(VoidSsaVariable),
            OpCodeTypes.Stelem_I2 => default(VoidSsaVariable),
            OpCodeTypes.Stelem_I4 => default(VoidSsaVariable),
            OpCodeTypes.Stelem_I8 => default(VoidSsaVariable),
            OpCodeTypes.Stelem_R4 => default(VoidSsaVariable),
            OpCodeTypes.Stelem_R8 => default(VoidSsaVariable),
            OpCodeTypes.Stelem_Ref => default(VoidSsaVariable),
            OpCodeTypes.Ldelem => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Stelem => default(VoidSsaVariable),
            OpCodeTypes.Unbox_Any => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Conv_Ovf_I1 => _nameGenerator.GetNextTemp(typeof(sbyte)),
            OpCodeTypes.Conv_Ovf_U1 => _nameGenerator.GetNextTemp(typeof(byte)),
            OpCodeTypes.Conv_Ovf_I2 => _nameGenerator.GetNextTemp(typeof(short)),
            OpCodeTypes.Conv_Ovf_U2 => _nameGenerator.GetNextTemp(typeof(ushort)),
            OpCodeTypes.Conv_Ovf_I4 => _nameGenerator.GetNextTemp(typeof(int)),
            OpCodeTypes.Conv_Ovf_U4 => _nameGenerator.GetNextTemp(typeof(uint)),
            OpCodeTypes.Conv_Ovf_I8 => _nameGenerator.GetNextTemp(typeof(long)),
            OpCodeTypes.Conv_Ovf_U8 => _nameGenerator.GetNextTemp(typeof(ulong)),
            OpCodeTypes.Refanyval => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ckfinite => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Mkrefany => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldtoken => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Conv_U2 => _nameGenerator.GetNextTemp(typeof(ushort)),
            OpCodeTypes.Conv_U1 => _nameGenerator.GetNextTemp(typeof(byte)),
            OpCodeTypes.Conv_I when node.Operand is MsilTypeOperand typeOperand => _nameGenerator.GetNextTemp(
                typeOperand.Value),
            OpCodeTypes.Conv_I => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Conv_Ovf_I when node.Operand is MsilTypeOperand typeOperand => _nameGenerator.GetNextTemp(
                typeOperand.Value),
            OpCodeTypes.Conv_Ovf_I => throw new InvalidOperationException("The type operand is not valid."),
            OpCodeTypes.Conv_Ovf_U when node.Operand is MsilTypeOperand typeOperand => _nameGenerator.GetNextTemp(
                typeOperand.Value),
            OpCodeTypes.Conv_Ovf_U => throw new InvalidOperationException("The type operand is not valid."),
            OpCodeTypes.Add_Ovf => _nameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Add_Ovf_Un => _nameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Mul_Ovf => _nameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Mul_Ovf_Un => _nameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Sub_Ovf => _nameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Sub_Ovf_Un => _nameGenerator.GetNextTemp(GetResultTypeFromBinaryArithmeticOperation(operands)),
            OpCodeTypes.Endfinally => default(VoidSsaVariable),
            OpCodeTypes.Leave => default(VoidSsaVariable),
            OpCodeTypes.Stind_I => default(VoidSsaVariable),
            OpCodeTypes.Conv_U => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Arglist => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ceq => _nameGenerator.GetNextTemp(typeof(bool)),
            OpCodeTypes.Cgt => _nameGenerator.GetNextTemp(typeof(bool)),
            OpCodeTypes.Cgt_Un => _nameGenerator.GetNextTemp(typeof(bool)),
            OpCodeTypes.Clt => _nameGenerator.GetNextTemp(typeof(bool)),
            OpCodeTypes.Clt_Un => _nameGenerator.GetNextTemp(typeof(bool)),
            OpCodeTypes.Ldftn => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldvirtftn => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldarg or OpCodeTypes.Ldarg_S when node.Operand is MsilInlineIntOperand operand =>
                _nameGenerator.GetNextTemp(_argumentSsaVariables[operand.Value].Type),
            OpCodeTypes.Ldarg or OpCodeTypes.Ldarg_S when node.Operand is MsilInlineVarOperand operand => _nameGenerator
                .GetNextTemp(operand.Value),
            OpCodeTypes.Ldarga => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Starg => default(VoidSsaVariable),
            OpCodeTypes.Ldloc or OpCodeTypes.Ldloc_S when node.Operand is MsilInlineIntOperand operand =>
                _nameGenerator.GetNextTemp(_localVariableSsaVariables[operand.Value].Type),
            OpCodeTypes.Ldloc or OpCodeTypes.Ldloc_S when node.Operand is MsilInlineVarOperand operand => _nameGenerator
                .GetNextTemp(operand.Value),
            OpCodeTypes.Ldloca or OpCodeTypes.Ldloca_S => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Stloc or OpCodeTypes.Stloc_S => default(VoidSsaVariable),
            OpCodeTypes.Localloc => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Endfilter => _nameGenerator.GetNextTemp(typeof(nint)),
        };
    }

    private ISsaVariable GetMethodCallReturnType(MsilMethodOperand operand)
    {
        var method = operand.MethodBase;

        if (method is MethodInfo methodInfo)
        {
            return new TempSsaVariable(_nameGenerator.NextLocalName(), methodInfo.ReturnType);
        }

        if (method is ConstructorInfo constructorInfo)
        {
            return new TempSsaVariable(
                _nameGenerator.NextLocalName(),
                constructorInfo.DeclaringType ?? throw new InvalidOperationException("The constructor does not have a declaring type."));
        }

        return default(VoidSsaVariable);
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
                    Parameter = _argumentSsaVariables[operand.Value].ToIrValue(),
                    Result = result.ToIrValue(),
                    MsilInstruction = node
                },
            OpCodeTypes.Ldarg or OpCodeTypes.Ldarg_S when node.Operand is MsilInlineVarOperand operand => new
                LoadArgumentInstruction
                {
                    Parameter = _argumentSsaVariables[operand.Index].ToIrValue(),
                    Result = result.ToIrValue(),
                    MsilInstruction = node
                },
            OpCodeTypes.Stloc or OpCodeTypes.Stloc_S when node.Operand is MsilInlineIntOperand operand => new
                StoreLocalInstruction
                {
                    Local = _localVariableSsaVariables[operand.Value].ToIrValue(),
                    Value = operands[0].ToIrValue(),
                    MsilInstruction = node
                },
            OpCodeTypes.Stloc or OpCodeTypes.Stloc_S when node.Operand is MsilInlineVarOperand operand => new
                StoreLocalInstruction
                {
                    Local = _localVariableSsaVariables[operand.Index].ToIrValue(),
                    Value = operands[0].ToIrValue(),
                    MsilInstruction = node
                },
            OpCodeTypes.Ldloc or OpCodeTypes.Ldloc_S when node.Operand is MsilInlineIntOperand operand => new
                LoadLocalInstruction
                {
                    Local = _localVariableSsaVariables[operand.Value].ToIrValue(),
                    Result = result.ToIrValue(),
                    MsilInstruction = node
                },
            OpCodeTypes.Ldloc or OpCodeTypes.Ldloc_S when node.Operand is MsilInlineVarOperand operand => new
                LoadLocalInstruction
                {
                    Local = _localVariableSsaVariables[operand.Index].ToIrValue(),
                    Result = result.ToIrValue(),
                    MsilInstruction = node
                },
            OpCodeTypes.Ldc_I4 or OpCodeTypes.Ldc_I4_S when node.Operand is MsilInlineIntOperand operand => new
                LoadConstantInstruction { Result = result.ToIrValue(), Value = operand.Value, MsilInstruction = node },
            OpCodeTypes.Ldc_I8 when node.Operand is MsilInlineLongOperand operand => new LoadConstantInstruction
            {
                Result = result.ToIrValue(), Value = operand.Value, MsilInstruction = node
            },
            OpCodeTypes.Ldc_R4 when node.Operand is MsilInlineSingleOperand operand => new LoadConstantInstruction
            {
                Result = result.ToIrValue(), Value = operand.Value, MsilInstruction = node
            },
            OpCodeTypes.Ldc_R8 when node.Operand is MsilInlineDoubleOperand operand => new LoadConstantInstruction
            {
                Result = result.ToIrValue(), Value = operand.Value, MsilInstruction = node
            },
            OpCodeTypes.Conv_I or OpCodeTypes.Conv_I1 or OpCodeTypes.Conv_I2 or OpCodeTypes.Conv_I4
                or OpCodeTypes.Conv_I8 or OpCodeTypes.Conv_R4 or OpCodeTypes.Conv_R8 or OpCodeTypes.Conv_U4
                or OpCodeTypes.Conv_U8 => new SignExtendInstruction
                {
                    Result = result.ToIrValue(), Value = operands[0].ToIrValue(), MsilInstruction = node
                },
            OpCodeTypes.Br or OpCodeTypes.Br_S when node.Operand is MsilBranchTargetOperand operand => new
                BranchInstruction { Target = FindBlockForOffset(basicBlocks, operand.Target), MsilInstruction = node },
            OpCodeTypes.Brtrue or OpCodeTypes.Brtrue_S when node.Operand is MsilBranchTargetOperand operand => new
                ConditionalBranchInstruction
                {
                    Condition = operands[0].ToIrValue(),
                    Target = FindBlockForOffset(basicBlocks, operand.Target),
                    FalseTarget = nextBlock,
                    MsilInstruction = node
                },
            OpCodeTypes.Brfalse or OpCodeTypes.Brfalse_S when node.Operand is MsilBranchTargetOperand operand => new
                ConditionalBranchInstruction
                {
                    Condition = operands[0].ToIrValue(),
                    Target = nextBlock,
                    FalseTarget = FindBlockForOffset(basicBlocks, operand.Target),
                    MsilInstruction = node
                },
            OpCodeTypes.Bge or OpCodeTypes.Bge_S when node.Operand is MsilBranchTargetOperand operand => new
                BranchGreaterThanInstruction
                {
                    Left = operands[0].ToIrValue(),
                    Right = operands[1].ToIrValue(),
                    Target = FindBlockForOffset(basicBlocks, operand.Target),
                    FalseTarget = nextBlock,
                    MsilInstruction = node
                },
            OpCodeTypes.Mul => new MultiplyInstruction
            {
                Result = result.ToIrValue(),
                Left = operands[0].ToIrValue(),
                Right = operands[1].ToIrValue(),
                MsilInstruction = node
            },
            OpCodeTypes.Add => new AddInstruction
            {
                Result = result.ToIrValue(),
                Left = operands[0].ToIrValue(),
                Right = operands[1].ToIrValue(),
                MsilInstruction = node
            },
            OpCodeTypes.Ldind_I or OpCodeTypes.Ldind_I1 or OpCodeTypes.Ldind_I2 or OpCodeTypes.Ldind_I4
                or OpCodeTypes.Ldind_I8 or OpCodeTypes.Ldind_U1 or OpCodeTypes.Ldind_U2 or OpCodeTypes.Ldind_U4
                or OpCodeTypes.Ldind_R4 or OpCodeTypes.Ldind_R8 =>
                new LoadElementFromPointerInstruction
                {
                    Result = result.ToIrValue(), Pointer = operands[0].ToIrValue(), MsilInstruction = node
                },
            OpCodeTypes.Stind_I or OpCodeTypes.Stind_I1 or OpCodeTypes.Stind_I2 or OpCodeTypes.Stind_I4
                or OpCodeTypes.Stind_I8 or OpCodeTypes.Stind_R4 or OpCodeTypes.Stind_R8 =>
                new StoreElementAtPointerInstruction
                {
                    Value = operands[0].ToIrValue(), Pointer = operands[1].ToIrValue(), MsilInstruction = node
                },
            OpCodeTypes.Clt => new CompareLessThanInstruction
            {
                Result = result.ToIrValue(),
                Left = operands[0].ToIrValue(),
                Right = operands[1].ToIrValue(),
                MsilInstruction = node
            },
            OpCodeTypes.Ret when operands.Count == 0 => new ReturnVoidInstruction { MsilInstruction = node },
            OpCodeTypes.Ret when operands.Count != 0 => new ReturnInstruction
            {
                Value = operands[0].ToIrValue(), MsilInstruction = node
            },
            OpCodeTypes.Call or OpCodeTypes.Calli or OpCodeTypes.Callvirt when node.Operand is MsilMethodOperand operand =>
                new CallInstruction
                {
                    MethodBase = operand.MethodBase ?? throw new InvalidOperationException("The method operand is null."),
                    Result = result.ToIrValue(),
                    Arguments = operands.Select(o => o.ToIrValue()).ToImmutableArray(),
                    MsilInstruction = node
                },
            OpCodeTypes.Ldloca or OpCodeTypes.Ldloca_S when node.Operand is MsilInlineIntOperand operand => new LoadLocalAddressInstruction
            {
                Result = result.ToIrValue(),
                Local = _localVariableSsaVariables[operand.Value].ToIrValue(),
                MsilInstruction = node
            },
            OpCodeTypes.Ldloca or OpCodeTypes.Ldloca_S when node.Operand is MsilInlineVarOperand operand => new LoadLocalAddressInstruction
            {
                Result = result.ToIrValue(),
                Local = _localVariableSsaVariables[operand.Index].ToIrValue(),
                MsilInstruction = node
            },
        };
    }
}