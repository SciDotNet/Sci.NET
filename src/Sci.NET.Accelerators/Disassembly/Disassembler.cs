// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Reflection.Emit;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Sci.NET.Accelerators.Disassembly.Instructions;
using Sci.NET.Accelerators.Disassembly.Instructions.Operands;

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
    private readonly Type[] _typeGenericArguments;
    private readonly Type[] _methodGenericArguments;
    private readonly Type[] _parameters;
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
        _parameters = method.GetParameters().Select(p => p.ParameterType).ToArray();
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
            ReflectedMethodBase = _method
        };
    }

    private List<Instruction<IOperand>> ReadInstructions()
    {
        var instructions = new List<Instruction<IOperand>>((_ilBytes.Length + 1) / 2);

        while (_blobReader.RemainingBytes > 0)
        {
            var offset = _blobReader.Offset;
            var opCode = ReadOpCode();
            var instruction = new Instruction<IOperand> { Offset = offset, OpCode = opCode, Size = _blobReader.Offset - offset, Operand = default(NoOperand) };

            if (opCode.OperandType != OperandType.InlineNone)
            {
                var operand = ReadOperand(instruction);

                instruction = new Instruction<IOperand> { Offset = offset, OpCode = opCode, Size = _blobReader.Offset - offset, Operand = operand };
            }

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
        switch (instruction.OpCode.OperandType)
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

                return new SwitchBranchesOperand { OperandType = instruction.OpCode.OperandType, Branches = branches, BaseOffset = baseOffset };
            case OperandType.ShortInlineBrTarget:
                return new BranchTargetOperand { OperandType = instruction.OpCode.OperandType, Target = _blobReader.ReadSByte() + _blobReader.Offset };
            case OperandType.InlineBrTarget:
                return new BranchTargetOperand { OperandType = instruction.OpCode.OperandType, Target = _blobReader.ReadInt32() + _blobReader.Offset };
            case OperandType.ShortInlineI:
                if (instruction.OpCode == OpCodes.Ldc_I4_S)
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
                return new MethodOperand { OperandType = instruction.OpCode.OperandType, MethodBase = methodBase };
            case OperandType.InlineField:
                var field = _module.ResolveField(_blobReader.ReadInt32(), _typeGenericArguments, _methodGenericArguments);
                return new FieldOperand { OperandType = instruction.OpCode.OperandType, FieldInfo = field };
            default:
                throw new NotSupportedException();
        }
    }
}