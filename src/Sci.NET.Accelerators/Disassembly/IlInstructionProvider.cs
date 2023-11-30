// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Reflection.Emit;

namespace Sci.NET.Accelerators.Disassembly;

/// <summary>
/// Provides a list of IL instructions.
/// </summary>
[PublicAPI]
public static class IlInstructionProvider
{
    private static readonly OpCode[] OneByteOpCodes = new OpCode[0x100];
    private static readonly OpCode[] TwoByteOpCodes = new OpCode[0x100];
    private static bool _initialised;

    /// <summary>
    /// Gets the opcode for the given value.
    /// </summary>
    /// <param name="opCode">The opcode value.</param>
    /// <returns>The opcode.</returns>
    public static OpCode GetOneByteOpCode(int opCode)
    {
        if (!_initialised)
        {
            Initialise();
        }

        return OneByteOpCodes[opCode];
    }

    /// <summary>
    /// Gets the opcode for the given value.
    /// </summary>
    /// <param name="opCode">The opcode value.</param>
    /// <returns>The opcode.</returns>
    public static OpCode GetTwoByteOpCode(int opCode)
    {
        if (!_initialised)
        {
            Initialise();
        }

        return TwoByteOpCodes[opCode];
    }

    private static void Initialise()
    {
        foreach (var field in typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            var opCode = (OpCode)field.GetValue(null) !;

            if (opCode.Size == 1)
            {
                OneByteOpCodes[(ushort)opCode.Value] = opCode;
            }
            else
            {
                var shiftedValue = (byte)(ushort)opCode.Value & 0xFF;
                TwoByteOpCodes[shiftedValue] = opCode;
            }
        }

        _initialised = true;
    }
}