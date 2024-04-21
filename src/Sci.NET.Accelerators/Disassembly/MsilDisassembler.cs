// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Reflection;
using System.Reflection.Emit;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Sci.NET.Accelerators.Disassembly.Operands;
using Sci.NET.Accelerators.Disassembly.Pdb;

namespace Sci.NET.Accelerators.Disassembly;

/// <summary>
/// Disassembles method bytes into MSIL instructions.
/// </summary>
internal class MsilDisassembler
{
    private readonly MsilMethodMetadata _methodMetadata;
    private readonly ImmutableDictionary<int, PdbSequencePoint> _sequencePoints;
    private readonly byte[] _ilBytes;
    private BlobReader _blobReader;

    /// <summary>
    /// Initializes a new instance of the <see cref="MsilDisassembler"/> class.
    /// </summary>
    /// <param name="metadata">The method to disassemble.</param>
    public unsafe MsilDisassembler(MsilMethodMetadata metadata)
    {
        _methodMetadata = metadata;
        _ilBytes = metadata.MethodBody.GetILAsByteArray() ?? throw new ArgumentException("Method body is null.", nameof(metadata));
        _blobReader = new BlobReader((byte*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_ilBytes)), _ilBytes.Length);
        _sequencePoints = metadata.MethodDebugInfo.SequencePoints.Select(x => new KeyValuePair<int, PdbSequencePoint>(x.Offset, x)).ToImmutableDictionary();
    }

    public static DisassembledMsilMethod DisassembleFromMethodBase(MethodBase methodBase)
    {
        var metadata = new MsilMethodMetadata(methodBase);
        var disassembler = new MsilDisassembler(metadata);
        return disassembler.Disassemble();
    }

    /// <summary>
    /// Disassembles the kernel into MSIL instructions.
    /// </summary>
    /// <returns>The disassembled method body.</returns>
    public DisassembledMsilMethod Disassemble()
    {
        var instructions = ReadInstructions();

        return new DisassembledMsilMethod { Metadata = _methodMetadata, Instructions = instructions.ToList() };
    }

    private static MsilInstruction<IMsilOperand> ExpandShortFormInstruction(MsilInstruction<IMsilOperand> instruction)
    {
        instruction = ExpandLdArgShortForm(instruction);
        instruction = ExpandLdLocShortForm(instruction);
        instruction = ExpandStLocShortForm(instruction);
        instruction = ExpandLdcI4ShortForm(instruction);

        return instruction;
    }

    private static MsilInstruction<IMsilOperand> ExpandLdArgShortForm(MsilInstruction<IMsilOperand> instruction)
    {
#pragma warning disable IDE0072
        return instruction.IlOpCode switch
#pragma warning restore IDE0072
        {
            OpCodeTypes.Ldarg_0 => instruction with { IlOpCode = OpCodeTypes.Ldarg, Name = "ldarg", Operand = new MsilInlineIntOperand { Value = 0 } },
            OpCodeTypes.Ldarg_1 => instruction with { IlOpCode = OpCodeTypes.Ldarg, Name = "ldarg", Operand = new MsilInlineIntOperand { Value = 1 } },
            OpCodeTypes.Ldarg_2 => instruction with { IlOpCode = OpCodeTypes.Ldarg, Name = "ldarg", Operand = new MsilInlineIntOperand { Value = 2 } },
            OpCodeTypes.Ldarg_3 => instruction with { IlOpCode = OpCodeTypes.Ldarg, Name = "ldarg", Operand = new MsilInlineIntOperand { Value = 3 } },
            _ => instruction
        };
    }

    private static MsilInstruction<IMsilOperand> ExpandLdLocShortForm(MsilInstruction<IMsilOperand> instruction)
    {
#pragma warning disable IDE0072
        return instruction.IlOpCode switch
#pragma warning restore IDE0072
        {
            OpCodeTypes.Ldloc_0 => instruction with { IlOpCode = OpCodeTypes.Ldloc, Name = "ldloc", Operand = new MsilInlineIntOperand { Value = 0 } },
            OpCodeTypes.Ldloc_1 => instruction with { IlOpCode = OpCodeTypes.Ldloc, Name = "ldloc", Operand = new MsilInlineIntOperand { Value = 1 } },
            OpCodeTypes.Ldloc_2 => instruction with { IlOpCode = OpCodeTypes.Ldloc, Name = "ldloc", Operand = new MsilInlineIntOperand { Value = 2 } },
            OpCodeTypes.Ldloc_3 => instruction with { IlOpCode = OpCodeTypes.Ldloc, Name = "ldloc", Operand = new MsilInlineIntOperand { Value = 3 } },
            _ => instruction
        };
    }

    private static MsilInstruction<IMsilOperand> ExpandStLocShortForm(MsilInstruction<IMsilOperand> instruction)
    {
#pragma warning disable IDE0072
        return instruction.IlOpCode switch
#pragma warning restore IDE0072
        {
            OpCodeTypes.Stloc_0 => instruction with { IlOpCode = OpCodeTypes.Stloc, Name = "stloc", Operand = new MsilInlineIntOperand { Value = 0 } },
            OpCodeTypes.Stloc_1 => instruction with { IlOpCode = OpCodeTypes.Stloc, Name = "stloc", Operand = new MsilInlineIntOperand { Value = 1 } },
            OpCodeTypes.Stloc_2 => instruction with { IlOpCode = OpCodeTypes.Stloc, Name = "stloc", Operand = new MsilInlineIntOperand { Value = 2 } },
            OpCodeTypes.Stloc_3 => instruction with { IlOpCode = OpCodeTypes.Stloc, Name = "stloc", Operand = new MsilInlineIntOperand { Value = 3 } },
            _ => instruction
        };
    }

    private static MsilInstruction<IMsilOperand> ExpandLdcI4ShortForm(MsilInstruction<IMsilOperand> instruction)
    {
#pragma warning disable IDE0072
        return instruction.IlOpCode switch
#pragma warning restore IDE0072
        {
            OpCodeTypes.Ldc_I4_0 => instruction with { IlOpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new MsilInlineIntOperand { Value = 0 } },
            OpCodeTypes.Ldc_I4_1 => instruction with { IlOpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new MsilInlineIntOperand { Value = 1 } },
            OpCodeTypes.Ldc_I4_2 => instruction with { IlOpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new MsilInlineIntOperand { Value = 2 } },
            OpCodeTypes.Ldc_I4_3 => instruction with { IlOpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new MsilInlineIntOperand { Value = 3 } },
            OpCodeTypes.Ldc_I4_4 => instruction with { IlOpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new MsilInlineIntOperand { Value = 4 } },
            OpCodeTypes.Ldc_I4_5 => instruction with { IlOpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new MsilInlineIntOperand { Value = 5 } },
            OpCodeTypes.Ldc_I4_6 => instruction with { IlOpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new MsilInlineIntOperand { Value = 6 } },
            OpCodeTypes.Ldc_I4_7 => instruction with { IlOpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new MsilInlineIntOperand { Value = 7 } },
            OpCodeTypes.Ldc_I4_8 => instruction with { IlOpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new MsilInlineIntOperand { Value = 8 } },
            OpCodeTypes.Ldc_I4_M1 => instruction with { IlOpCode = OpCodeTypes.Ldc_I4, Name = "ldc.i4", Operand = new MsilInlineIntOperand { Value = -1 } },
            _ => instruction
        };
    }

    private List<MsilInstruction<IMsilOperand>> ReadInstructions()
    {
        var instructions = new List<MsilInstruction<IMsilOperand>>((_ilBytes.Length + 1) / 2);

        for (var index = 0; _blobReader.RemainingBytes > 0; index++)
        {
            var offset = _blobReader.Offset;
            var opCode = ReadOpCode();
            var instruction = new MsilInstruction<IMsilOperand>(
                opCode,
                offset,
                opCode.Size,
                index);

            if (opCode.OperandType != OperandType.InlineNone)
            {
                var operand = ReadOperand(instruction);
                instruction = new MsilInstruction<IMsilOperand>(
                    opCode,
                    offset,
                    _blobReader.Offset - offset,
                    index);
                instruction = instruction with { Operand = operand };
            }
            else
            {
                instruction = instruction with { Operand = default(MsilNoOperand), OperandType = OperandType.InlineNone };
            }

            if (_sequencePoints.TryGetValue(instruction.Offset, out var value))
            {
                instruction = instruction with { SequencePoint = value };
            }

            instruction = ExpandShortFormInstruction(instruction);
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

    private IMsilOperand ReadOperand(MsilInstruction<IMsilOperand> instruction)
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

                return new MsilSwitchTargetsOperand { OperandType = instruction.OperandType, Branches = branches, BaseOffset = baseOffset };
            case OperandType.ShortInlineBrTarget:
                return new MsilBranchTargetOperand { OperandType = instruction.OperandType, Target = _blobReader.ReadSByte() + _blobReader.Offset };
            case OperandType.InlineBrTarget:
                return new MsilBranchTargetOperand { OperandType = instruction.OperandType, Target = _blobReader.ReadInt32() + _blobReader.Offset };
            case OperandType.ShortInlineI:
                if (instruction.IlOpCode == OpCodeTypes.Ldc_I4_S)
                {
                    return new MsilInlineSByteOperand { Value = _blobReader.ReadSByte() };
                }

                return new MsilInlineByteOperand { Value = _blobReader.ReadByte() };
            case OperandType.InlineI:
                return new MsilInlineIntOperand { Value = _blobReader.ReadInt32() };
            case OperandType.ShortInlineR:
                return new MsilInlineSingleOperand { Value = _blobReader.ReadSingle() };
            case OperandType.InlineR:
                return new MsilInlineDoubleOperand { Value = _blobReader.ReadDouble() };
            case OperandType.InlineI8:
                return new MsilInlineLongOperand { Value = _blobReader.ReadInt64() };
            case OperandType.ShortInlineVar:
                return ResolveInlineVarOperand(instruction, _blobReader.ReadByte());
            case OperandType.InlineVar:
                return ResolveInlineVarOperand(instruction, _blobReader.ReadInt16());
            case OperandType.InlineSig:
                return new MemberInfoOperand { Value = _methodMetadata.Module.ResolveMember(_blobReader.ReadInt32()), OperandType = OperandType.InlineSig };
            case OperandType.InlineString:
                return new MsilInlineStringOperand { Value = _methodMetadata.Module.ResolveString(_blobReader.ReadInt32()) };
            case OperandType.InlineTok:
                return new MemberInfoOperand { Value = _methodMetadata.Module.ResolveMember(_blobReader.ReadInt32(), _methodMetadata.TypeGenericArguments.ToArray(), _methodMetadata.MethodGenericArguments.ToArray()), OperandType = OperandType.InlineTok };
            case OperandType.InlineType:
                return new MsilTypeOperand { Value = _methodMetadata.Module.ResolveType(_blobReader.ReadInt32(), _methodMetadata.TypeGenericArguments.ToArray(), _methodMetadata.MethodGenericArguments.ToArray()), OperandType = OperandType.InlineType };
            case OperandType.InlineMethod:
                var methodBase = _methodMetadata.Module.ResolveMethod(_blobReader.ReadInt32(), _methodMetadata.TypeGenericArguments.ToArray(), _methodMetadata.MethodGenericArguments.ToArray());
                return new MsilMethodOperand { OperandType = instruction.OperandType, MethodBase = methodBase };
            case OperandType.InlineField:
                var field = _methodMetadata.Module.ResolveField(_blobReader.ReadInt32(), _methodMetadata.TypeGenericArguments.ToArray(), _methodMetadata.MethodGenericArguments.ToArray());
                return new MsilFieldOperand { OperandType = instruction.OperandType, FieldInfo = field };
            default:
                throw new NotSupportedException();
        }
    }

    private MsilInlineVarOperand ResolveInlineVarOperand(MsilInstruction<IMsilOperand> instruction, int index)
    {
        var isArgument = instruction.IlOpCode is OpCodeTypes.Ldarg
            or OpCodeTypes.Ldarg_0
            or OpCodeTypes.Ldarg_1
            or OpCodeTypes.Ldarg_2
            or OpCodeTypes.Ldarg_3
            or OpCodeTypes.Ldarg_S
            or OpCodeTypes.Ldarga
            or OpCodeTypes.Ldarga_S
            or OpCodeTypes.Starg
            or OpCodeTypes.Starg_S;

        return isArgument
            ? new MsilInlineVarOperand { Value = _methodMetadata.Parameters[index].ParameterType, Index = index, OperandType = OperandType.InlineVar, Name = _methodMetadata.Parameters[index].Name ?? $"param_{index}" }
            : new MsilInlineVarOperand { Value = _methodMetadata.Variables[index].Type, Index = index, OperandType = OperandType.InlineVar, Name = _methodMetadata.Variables[index].Name };
    }
}