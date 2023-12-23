// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Reflection.Emit;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Sci.NET.Accelerators.Disassembly.Operands;

namespace Sci.NET.Accelerators.Disassembly;

/// <summary>
/// Disassembles a kernel into MSIL instructions.
/// </summary>
[PublicAPI]
public class Disassembler
{
    private readonly MethodBody _methodBody;
    private readonly MethodBase _method;
    private readonly Module _module;
    private readonly ParameterInfo[] _parameters;
    private readonly Type[] _typeGenericArguments;
    private readonly Type[] _methodGenericArguments;
    private readonly byte[] _ilBytes;
    private BlobReader _blobReader;

    /// <summary>
    /// Initializes a new instance of the <see cref="Disassembler"/> class.
    /// </summary>
    /// <param name="method">The method to disassemble.</param>
    public unsafe Disassembler(MethodBase method)
    {
        _methodBody = method.GetMethodBody() ?? throw new InvalidOperationException("Method body is empty.");
        _ilBytes = _methodBody.GetILAsByteArray() ?? throw new InvalidOperationException("Method body is empty.");
        _blobReader = new BlobReader((byte*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_ilBytes)), _ilBytes.Length);
        _module = method.Module;
        _typeGenericArguments = method.DeclaringType?.GetGenericArguments() ?? Array.Empty<Type>();
        _methodGenericArguments = method.IsGenericMethod ? method.GetGenericArguments() : Array.Empty<Type>();
        _parameters = method.GetParameters().ToArray();
        _method = method;
    }

    /// <summary>
    /// Disassembles the kernel into MSIL instructions.
    /// </summary>
    /// <returns>The disassembled method body.</returns>
    public DisassembledMethod Disassemble()
    {
        return new DisassembledMethod
        {
            MaxStack = _methodBody.MaxStackSize,
            CodeSize = _ilBytes.Length,
            InitLocals = _methodBody.InitLocals,
            LocalVariablesSignatureToken = _methodBody.LocalSignatureMetadataToken,
            Variables = _methodBody.LocalVariables,
            Instructions = ReadInstructions().ToList(),
            TypeGenericArguments = _typeGenericArguments.ToList(),
            MethodGenericArguments = _methodGenericArguments.ToList(),
            Parameters = _parameters.ToList(),
            ReflectedMethodBase = _method,
            ReturnType = _method is MethodInfo methodInfo ? methodInfo.ReturnType : typeof(void)
        };
    }

    private static Instruction<IOperand> ExpandMacro(Instruction<IOperand> instruction)
    {
        instruction = ConvertLdArgMacro(instruction);
        instruction = ConvertLdLocMacro(instruction);
        instruction = ConvertStLocMacro(instruction);
        instruction = ConvertLdcI4Macro(instruction);

        return instruction;
    }

    private static Instruction<IOperand> ConvertLdArgMacro(Instruction<IOperand> instruction)
    {
#pragma warning disable IDE0072
        return instruction.OpCode switch
#pragma warning restore IDE0072
        {
            OpCodeTypes.Ldarg_0 => instruction with { OpCode = OpCodeTypes.Ldarg, Name = "ldarg", Operand = new InlineIntOperand { Value = 0 } },
            OpCodeTypes.Ldarg_1 => instruction with { OpCode = OpCodeTypes.Ldarg, Name = "ldarg", Operand = new InlineIntOperand { Value = 1 } },
            OpCodeTypes.Ldarg_2 => instruction with { OpCode = OpCodeTypes.Ldarg, Name = "ldarg", Operand = new InlineIntOperand { Value = 2 } },
            OpCodeTypes.Ldarg_3 => instruction with { OpCode = OpCodeTypes.Ldarg, Name = "ldarg", Operand = new InlineIntOperand { Value = 3 } },
            _ => instruction
        };
    }

    private static Instruction<IOperand> ConvertLdLocMacro(Instruction<IOperand> instruction)
    {
#pragma warning disable IDE0072
        return instruction.OpCode switch
#pragma warning restore IDE0072
        {
            OpCodeTypes.Ldloc_0 => instruction with { OpCode = OpCodeTypes.Ldloc, Name = "ldloc", Operand = new InlineIntOperand { Value = 0 } },
            OpCodeTypes.Ldloc_1 => instruction with { OpCode = OpCodeTypes.Ldloc, Name = "ldloc", Operand = new InlineIntOperand { Value = 1 } },
            OpCodeTypes.Ldloc_2 => instruction with { OpCode = OpCodeTypes.Ldloc, Name = "ldloc", Operand = new InlineIntOperand { Value = 2 } },
            OpCodeTypes.Ldloc_3 => instruction with { OpCode = OpCodeTypes.Ldloc, Name = "ldloc", Operand = new InlineIntOperand { Value = 3 } },
            _ => instruction
        };
    }

    private static Instruction<IOperand> ConvertStLocMacro(Instruction<IOperand> instruction)
    {
#pragma warning disable IDE0072
        return instruction.OpCode switch
#pragma warning restore IDE0072
        {
            OpCodeTypes.Stloc_0 => instruction with { OpCode = OpCodeTypes.Stloc, Name = "stloc", Operand = new InlineIntOperand { Value = 0 } },
            OpCodeTypes.Stloc_1 => instruction with { OpCode = OpCodeTypes.Stloc, Name = "stloc", Operand = new InlineIntOperand { Value = 1 } },
            OpCodeTypes.Stloc_2 => instruction with { OpCode = OpCodeTypes.Stloc, Name = "stloc", Operand = new InlineIntOperand { Value = 2 } },
            OpCodeTypes.Stloc_3 => instruction with { OpCode = OpCodeTypes.Stloc, Name = "stloc", Operand = new InlineIntOperand { Value = 3 } },
            _ => instruction
        };
    }

    private static Instruction<IOperand> ConvertLdcI4Macro(Instruction<IOperand> instruction)
    {
#pragma warning disable IDE0072
        return instruction.OpCode switch
#pragma warning restore IDE0072
        {
            OpCodeTypes.Ldc_I4_0 => instruction with { OpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new InlineIntOperand { Value = 0 } },
            OpCodeTypes.Ldc_I4_1 => instruction with { OpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new InlineIntOperand { Value = 1 } },
            OpCodeTypes.Ldc_I4_2 => instruction with { OpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new InlineIntOperand { Value = 2 } },
            OpCodeTypes.Ldc_I4_3 => instruction with { OpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new InlineIntOperand { Value = 3 } },
            OpCodeTypes.Ldc_I4_4 => instruction with { OpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new InlineIntOperand { Value = 4 } },
            OpCodeTypes.Ldc_I4_5 => instruction with { OpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new InlineIntOperand { Value = 5 } },
            OpCodeTypes.Ldc_I4_6 => instruction with { OpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new InlineIntOperand { Value = 6 } },
            OpCodeTypes.Ldc_I4_7 => instruction with { OpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new InlineIntOperand { Value = 7 } },
            OpCodeTypes.Ldc_I4_8 => instruction with { OpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new InlineIntOperand { Value = 8 } },
            OpCodeTypes.Ldc_I4_M1 => instruction with { OpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new InlineIntOperand { Value = -1 } },
            _ => instruction
        };
    }

    private List<Instruction<IOperand>> ReadInstructions()
    {
        var instructions = new List<Instruction<IOperand>>((_ilBytes.Length + 1) / 2);

        for (var index = 0; _blobReader.RemainingBytes > 0; index++)
        {
            var offset = _blobReader.Offset;
            var opCode = ReadOpCode();
            var instruction = new Instruction<IOperand>(
                opCode,
                offset,
                opCode.Size,
                index);

            if (opCode.OperandType != OperandType.InlineNone)
            {
                var operand = ReadOperand(instruction);
                instruction = new Instruction<IOperand>(
                    opCode,
                    offset,
                    _blobReader.Offset - offset,
                    index);
                instruction = instruction with { Operand = operand };
            }

            instruction = ExpandMacro(instruction);
            instructions.Add(instruction);
        }

        return instructions;
    }

    private OpCode ReadOpCode()
    {
        int opCode = _blobReader.ReadByte();

        if (opCode == 0xfe)
        {
            var twoByteOpCode = _blobReader.ReadByte();
            return IlInstructionProvider.GetTwoByteOpCode(twoByteOpCode);
        }

        return IlInstructionProvider.GetOneByteOpCode(opCode);
    }

    private IOperand ReadOperand(Instruction<IOperand> instruction)
    {
#pragma warning disable IDE0010
        switch (instruction.OperandType)
#pragma warning restore IDE0010
        {
            case OperandType.InlineSwitch:
                var length = _blobReader.ReadInt32();
                var baseOffset = _blobReader.Offset + (4 * length);
                var branches = new int[length];

                for (var i = 0; i < length; i++)
                {
                    branches[i] = baseOffset + _blobReader.ReadInt32();
                }

                return new SwitchBranchesOperand { OperandType = instruction.OperandType, Branches = branches, BaseOffset = baseOffset };
            case OperandType.ShortInlineBrTarget:
                return new BranchTargetOperand { OperandType = instruction.OperandType, Target = _blobReader.ReadSByte() + _blobReader.Offset };
            case OperandType.InlineBrTarget:
                return new BranchTargetOperand { OperandType = instruction.OperandType, Target = _blobReader.ReadInt32() + _blobReader.Offset };
            case OperandType.ShortInlineI:
                if (instruction.OpCode == OpCodeTypes.Ldc_I4_S)
                {
                    return new InlineSbyteOperand { Value = _blobReader.ReadSByte() };
                }

                return new InlineByteOperand { Value = _blobReader.ReadByte() };
            case OperandType.InlineI:
                return new InlineIntOperand { Value = _blobReader.ReadInt32() };
            case OperandType.ShortInlineR:
                return new InlineSingleOperand { Value = _blobReader.ReadSingle() };
            case OperandType.InlineR:
                return new InlineDoubleOperand { Value = _blobReader.ReadDouble() };
            case OperandType.InlineI8:
                return new InlineLongOperand { Value = _blobReader.ReadInt64() };
            case OperandType.ShortInlineVar:
                return new InlineVarOperand { Value = _methodBody.LocalVariables[_blobReader.ReadByte()], OperandType = OperandType.ShortInlineVar };
            case OperandType.InlineVar:
                return new InlineVarOperand { Value = _methodBody.LocalVariables[_blobReader.ReadUInt16()], OperandType = OperandType.InlineVar };
            case OperandType.InlineSig:
                return new MemberInfoOperand { Value = _module.ResolveMember(_blobReader.ReadInt32()), OperandType = OperandType.InlineSig };
            case OperandType.InlineString:
                return new InlineStringOperand { Value = _module.ResolveString(_blobReader.ReadInt32()) };
            case OperandType.InlineTok:
                return new MemberInfoOperand { Value = _module.ResolveMember(_blobReader.ReadInt32(), _typeGenericArguments, _methodGenericArguments), OperandType = OperandType.InlineTok };
            case OperandType.InlineType:
                return new TypeOperand { Value = _module.ResolveType(_blobReader.ReadInt32(), _typeGenericArguments, _methodGenericArguments), OperandType = OperandType.InlineType };
            case OperandType.InlineMethod:
                var methodBase = _module.ResolveMethod(_blobReader.ReadInt32(), _typeGenericArguments, _methodGenericArguments);
                return new MethodOperand { OperandType = instruction.OperandType, MethodBase = methodBase };
            case OperandType.InlineField:
                var field = _module.ResolveField(_blobReader.ReadInt32(), _typeGenericArguments, _methodGenericArguments);
                return new FieldOperand { OperandType = instruction.OperandType, FieldInfo = field };
            default:
                throw new NotSupportedException();
        }
    }
}