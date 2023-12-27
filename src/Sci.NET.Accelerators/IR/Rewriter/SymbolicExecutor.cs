// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection;
using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Cfg;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.IR.Rewriter.Variables;

namespace Sci.NET.Accelerators.IR.Rewriter;

/// <summary>
/// Represents a new symbolic executor.
/// </summary>
[PublicAPI]
public class SymbolicExecutor
{
#pragma warning disable SA1010, SA1009
    private static readonly Type[] TypePrecedenceBitManipulation = [typeof(ulong), typeof(long), typeof(uint), typeof(int), typeof(ushort), typeof(short), typeof(byte), typeof(sbyte)];
    private static readonly Type[] TypePrecedenceArithmetic = [typeof(double), typeof(float), typeof(ulong), typeof(long), typeof(uint), typeof(int), typeof(ushort), typeof(short), typeof(byte), typeof(sbyte)];
#pragma warning restore SA1009, SA1010

    private readonly MsilControlFlowGraph _cfg;
    private readonly DisassembledMethod _disassembledMethod;
    private readonly LocalVariableSsaVariable[] _localVariableSsaVariables;
    private readonly ArgumentSsaVariable[] _argumentSsaVariables;
    private readonly VariableNameGenerator _nameGenerator;
    private readonly Stack<ISsaVariable> _stack;

    /// <summary>
    /// Initializes a new instance of the <see cref="SymbolicExecutor"/> class.
    /// </summary>
    /// <param name="cfg">The control flow graph.</param>
    /// <param name="disassembledMethod">The disassembled method.</param>
    public SymbolicExecutor(MsilControlFlowGraph cfg, DisassembledMethod disassembledMethod)
    {
        _cfg = cfg;
        _disassembledMethod = disassembledMethod;
        _nameGenerator = new VariableNameGenerator();
        _localVariableSsaVariables = new LocalVariableSsaVariable[disassembledMethod.Variables.Count];
        _stack = new Stack<ISsaVariable>();

        for (var index = 0; index < disassembledMethod.Variables.Count; index++)
        {
            _localVariableSsaVariables[index] = new LocalVariableSsaVariable(index, _nameGenerator.NextLocalName(), disassembledMethod.Variables[index].LocalType);
        }

        _argumentSsaVariables = new ArgumentSsaVariable[disassembledMethod.Parameters.Count];

        for (var index = 0; index < disassembledMethod.Parameters.Count; index++)
        {
            _argumentSsaVariables[index] = new ArgumentSsaVariable(index, $"arg_{index}", disassembledMethod.Parameters[index].ParameterType);
        }
    }

    /// <summary>
    /// Executes the symbolic execution.
    /// </summary>
    /// <returns>The symbols.</returns>
    public SsaMethod Execute()
    {
        var symbols = new List<SsaInstruction>();

        foreach (var node in _cfg.Nodes)
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

            if (result is not VoidSsaVariable)
            {
                _stack.Push(result);
            }

            symbols.Add(
                new SsaInstruction
                {
                    Operands = operands,
                    IlOpCode = node.Instruction.IlOpCode,
                    Result = result,
                    IsLeader = node.IsLeader,
                    IsTerminator = node.IsTerminator,
                    Offset = node.Instruction.Offset,
                    NextInstructionIndices = node.NextInstructions.Select(x => x.Index).ToList(),
                    MsilInstruction = node.Instruction
                });
        }

        return new SsaMethod(symbols);
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

    private static int GetMethodCallPopBehaviour(MethodOperand operand)
    {
        var method = operand.MethodBase;

        if (method is not MethodInfo methodInfo)
        {
            throw new InvalidOperationException("The method operand does not have a method base.");
        }

        var popCount = methodInfo.GetParameters().Length;

#pragma warning disable IDE0010
        switch (methodInfo.CallingConvention)
#pragma warning restore IDE0010
        {
            case CallingConventions.VarArgs:
                throw new NotSupportedException("Calling methods with variable number of arguments is not yet supported.");
            case CallingConventions.HasThis:
                popCount++;
                break;
        }

        if (methodInfo.ReturnType != typeof(void))
        {
            popCount++;
        }

        return popCount;
    }

    private List<ISsaVariable> ExtractOperands(IMsilControlFlowGraphNode node)
    {
        var operands = new List<ISsaVariable>();
#pragma warning disable IDE0010
        switch (node.Instruction.IlOpCode)
#pragma warning restore IDE0010
        {
            case OpCodeTypes.Ldarg or OpCodeTypes.Ldarg_S when node.Instruction.Operand is InlineIntOperand index:
                operands.Add(_argumentSsaVariables[index.Value]);
                break;
            case OpCodeTypes.Ldloc or OpCodeTypes.Ldloc_S when node.Instruction.Operand is InlineIntOperand index:
                operands.Add(_localVariableSsaVariables[index.Value]);
                break;
            case OpCodeTypes.Ldc_I4 or OpCodeTypes.Ldc_I4_S when node.Instruction.Operand is InlineIntOperand value:
                operands.Add(new Int4ConstantSsaVariable(_nameGenerator.NextTemporaryName(), value.Value));
                break;
            case OpCodeTypes.Ldc_I8 when node.Instruction.Operand is InlineLongOperand value:
                operands.Add(new Int8ConstantSsaVariable(_nameGenerator.NextTemporaryName(), value.Value));
                break;
            case OpCodeTypes.Ldc_R4 when node.Instruction.Operand is InlineSingleOperand value:
                operands.Add(new Float4ConstantSsaVariable(_nameGenerator.NextTemporaryName(), value.Value));
                break;
            case OpCodeTypes.Ldc_R8 when node.Instruction.Operand is InlineDoubleOperand value:
                operands.Add(new Float8ConstantSsaVariable(_nameGenerator.NextTemporaryName(), value.Value));
                break;
        }

        return operands;
    }

    private int GetPopCount(IMsilControlFlowGraphNode node)
    {
        return node.Instruction.PopBehaviour switch
        {
            PopBehaviour.None => 0,
            PopBehaviour.Pop1 or PopBehaviour.Popi or PopBehaviour.Popref => 1,
            PopBehaviour.Pop1_pop1 or PopBehaviour.Popi_popi or PopBehaviour.Popi_popi8 or PopBehaviour.Popi_popr4 or PopBehaviour.Popi_popr8 or PopBehaviour.Popref_pop1 or PopBehaviour.Popi_pop1 or PopBehaviour.Popref_popi => 2,
            PopBehaviour.Popi_popi_popi or PopBehaviour.Popref_popi_popi or PopBehaviour.Popref_popi_popi8 or PopBehaviour.Popref_popi_popr4 or PopBehaviour.Popref_popi_popr8 or PopBehaviour.Popref_popi_popref or PopBehaviour.Popref_popi_pop1 => 3,
            PopBehaviour.Varpop when node.Instruction.IlOpCode is OpCodeTypes.Ret && _stack.Count == 1 => 1,
            PopBehaviour.Varpop when node.Instruction.IlOpCode is OpCodeTypes.Ret && _stack.Count == 0 && _disassembledMethod.ReturnType == typeof(void) => 0,
            PopBehaviour.Varpop when node.Instruction.IlOpCode is OpCodeTypes.Ret && _stack.Count > 0 => throw new InvalidOperationException("The stack is not empty."),
            PopBehaviour.Varpop when node.Instruction.Operand is MethodOperand operand => GetMethodCallPopBehaviour(operand),
            PopBehaviour.Varpop => throw new NotSupportedException("Varpop is not supported."),
            _ => throw new InvalidOperationException("The pop behaviour is not supported.")
        };
    }

    private ISsaVariable GetResult(IMsilControlFlowGraphNode node, List<ISsaVariable> operands)
    {
#pragma warning disable IDE0072
        return node.Instruction.IlOpCode switch
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
            OpCodeTypes.Call or OpCodeTypes.Callvirt => GetMethodCallReturnType(node.Instruction.Operand is MethodOperand operand ? operand : default),
            OpCodeTypes.Calli => GetMethodCallReturnType(node.Instruction.Operand is MethodOperand operand ? operand : default),
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
            OpCodeTypes.Ldobj when node.Instruction.Operand is TypeOperand typeOperand => _nameGenerator.GetNextTemp(typeOperand.Value),
            OpCodeTypes.Ldobj => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Ldstr => _nameGenerator.GetNextTemp(typeof(string)),
            OpCodeTypes.Newobj => GetMethodCallReturnType(node.Instruction.Operand is MethodOperand operand ? operand : default),
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
            OpCodeTypes.Conv_Ovf_I_Un when node.Instruction.Operand is TypeOperand type => _nameGenerator.GetNextTemp(type.Value),
            OpCodeTypes.Conv_Ovf_I_Un => throw new InvalidOperationException("The type operand is not valid."),
            OpCodeTypes.Conv_Ovf_U_Un when node.Instruction.Operand is TypeOperand type => _nameGenerator.GetNextTemp(type.Value),
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
            OpCodeTypes.Ldelem_I when node.Instruction.Operand is TypeOperand typeOperand => _nameGenerator.GetNextTemp(typeOperand.Value),
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
            OpCodeTypes.Conv_I when node.Instruction.Operand is TypeOperand typeOperand => _nameGenerator.GetNextTemp(typeOperand.Value),
            OpCodeTypes.Conv_I => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Conv_Ovf_I when node.Instruction.Operand is TypeOperand typeOperand => _nameGenerator.GetNextTemp(typeOperand.Value),
            OpCodeTypes.Conv_Ovf_I => throw new InvalidOperationException("The type operand is not valid."),
            OpCodeTypes.Conv_Ovf_U when node.Instruction.Operand is TypeOperand typeOperand => _nameGenerator.GetNextTemp(typeOperand.Value),
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
            OpCodeTypes.Ldarg or OpCodeTypes.Ldarg_S when node.Instruction.Operand is InlineIntOperand operand => _argumentSsaVariables[operand.Value],
            OpCodeTypes.Ldarg or OpCodeTypes.Ldarg_S when node.Instruction.Operand is InlineVarOperand operand => _nameGenerator.GetNextTemp(operand.Value),
            OpCodeTypes.Ldarga => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Starg => default(VoidSsaVariable),
            OpCodeTypes.Ldloc or OpCodeTypes.Ldloc_S when node.Instruction.Operand is InlineIntOperand operand => _localVariableSsaVariables[operand.Value],
            OpCodeTypes.Ldloc or OpCodeTypes.Ldloc_S when node.Instruction.Operand is InlineVarOperand operand => _nameGenerator.GetNextTemp(operand.Value),
            OpCodeTypes.Ldloca => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Stloc or OpCodeTypes.Stloc_S => default(VoidSsaVariable),
            OpCodeTypes.Localloc => _nameGenerator.GetNextTemp(typeof(nint)),
            OpCodeTypes.Endfilter => _nameGenerator.GetNextTemp(typeof(nint)),
        };
    }

    private ISsaVariable GetMethodCallReturnType(MethodOperand operand)
    {
        var method = operand.MethodBase;

        if (method is MethodInfo methodInfo)
        {
            return new TempSsaVariable(_nameGenerator.NextLocalName(), methodInfo.ReturnType);
        }

        return default(VoidSsaVariable);
    }
}