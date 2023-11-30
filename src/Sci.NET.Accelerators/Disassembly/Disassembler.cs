// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Reflection.Emit;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Sci.NET.Accelerators.Disassembly;

/// <summary>
/// Disassembles a kernel into MSIL instructions.
/// </summary>
[PublicAPI]
public class Disassembler
{
    private readonly MethodBase _method;
    private readonly MethodBody _methodBody;
    private readonly byte[] _ilBytes;
    private BlobReader _blobReader;

    /// <summary>
    /// Initializes a new instance of the <see cref="Disassembler"/> class.
    /// </summary>
    /// <param name="method">The method to disassemble.</param>
    public unsafe Disassembler(MethodBase method)
    {
        _method = method;
        _methodBody = method.GetMethodBody() ?? throw new InvalidOperationException("Method body is empty.");
        _ilBytes = _methodBody.GetILAsByteArray() ?? throw new InvalidOperationException("Method body is empty.");
        _blobReader = new BlobReader((byte*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_ilBytes)), _ilBytes.Length);
    }

    /// <summary>
    /// Disassembles the kernel into MSIL instructions.
    /// </summary>
    /// <returns>The disassembled method body.</returns>
    public DisassembledMethodBody Disassemble()
    {
        var typeGenericArguments = new List<Type>();
        var methodGenericArguments = new List<Type>();

        if (_method.DeclaringType?.IsGenericType ?? false)
        {
            typeGenericArguments.AddRange(_method.DeclaringType.GetGenericArguments());
        }

        if (_method.IsGenericMethod)
        {
            methodGenericArguments.AddRange(_method.GetGenericArguments());
        }

        return new DisassembledMethodBody
        {
            MaxStack = _methodBody.MaxStackSize,
            CodeSize = _ilBytes.Length,
            InitLocals = _methodBody.InitLocals,
            LocalVariablesSignatureToken = _methodBody.LocalSignatureMetadataToken,
            Variables = _methodBody.LocalVariables,
            Instructions = ReadInstructions(),
            TypeGenericArguments = typeGenericArguments,
            MethodGenericArguments = methodGenericArguments
        };
    }

    private List<Instruction> ReadInstructions()
    {
        var instructions = new List<Instruction>((_ilBytes.Length + 1) / 2);

        while (_blobReader.RemainingBytes > 0)
        {
            var offset = _blobReader.Offset;
            var opCode = ReadOpCode();
            var instruction = new Instruction(offset, opCode);

            if (opCode.OperandType != OperandType.InlineNone)
            {
                instruction.Operand = ReadOperand(instruction);
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

    private object ReadOperand(Instruction instruction)
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

                return branches;
            case OperandType.ShortInlineBrTarget:
                return _blobReader.ReadSByte() + _blobReader.Offset;
            case OperandType.InlineBrTarget:
                return _blobReader.ReadInt32() + _blobReader.Offset;
            case OperandType.ShortInlineI:
                if (instruction.OpCode == OpCodes.Ldc_I4_S)
                {
                    return _blobReader.ReadSByte();
                }

                return _blobReader.ReadByte();
            case OperandType.InlineI:
                return _blobReader.ReadInt32();
            case OperandType.ShortInlineR:
                return _blobReader.ReadSingle();
            case OperandType.InlineR:
                return _blobReader.ReadDouble();
            case OperandType.InlineI8:
                return _blobReader.ReadInt64();
            case OperandType.ShortInlineVar:
                return _methodBody.LocalVariables[_blobReader.ReadByte()];
            case OperandType.InlineVar:
                return _methodBody.LocalVariables[_blobReader.ReadUInt16()];
            case OperandType.InlineSig:
                return _blobReader.ReadBlobHandle();
            case OperandType.InlineString:
                return _blobReader.ReadInt32();
            case OperandType.InlineTok:
            case OperandType.InlineType:
            case OperandType.InlineMethod:
            case OperandType.InlineField:
                return _blobReader.ReadUInt32();
            default:
                throw new NotSupportedException();
        }
    }
}