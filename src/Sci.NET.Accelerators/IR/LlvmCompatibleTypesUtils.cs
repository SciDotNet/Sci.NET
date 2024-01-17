// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Numerics;

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// Represents the possible LLVM types.
/// </summary>
[PublicAPI]
public static class LlvmCompatibleTypesUtils
{
    /// <summary>
    /// Converts the <see cref="LlvmCompatibleTypes"/> to a string.
    /// </summary>
    /// <param name="compatibleType">The <see cref="LlvmCompatibleTypes"/> to convert.</param>
    /// <returns>The string representation of the <see cref="LlvmCompatibleTypes"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if the <see cref="LlvmCompatibleTypes"/> is not recognized.</exception>
    public static string GetCompilerString(this LlvmCompatibleTypes compatibleType)
    {
        return compatibleType switch
        {
            LlvmCompatibleTypes.Void => "void",
            LlvmCompatibleTypes.Pointer => "pointer",
            LlvmCompatibleTypes.Fp16 => "fp16",
            LlvmCompatibleTypes.Bf16 => "bf16",
            LlvmCompatibleTypes.Fp32 => "single",
            LlvmCompatibleTypes.Fp64 => "double",
            LlvmCompatibleTypes.I1 => "i1",
            LlvmCompatibleTypes.I8 => "i8",
            LlvmCompatibleTypes.I16 => "i16",
            LlvmCompatibleTypes.I32 => "i32",
            LlvmCompatibleTypes.I64 => "i64",
            LlvmCompatibleTypes.I128 => "i128",
            _ => throw new ArgumentOutOfRangeException(nameof(compatibleType))
        };
    }

    /// <summary>
    /// Converts the <see cref="Type"/> to a <see cref="LlvmCompatibleTypes"/>.
    /// </summary>
    /// <param name="type">The <see cref="Type"/> to convert.</param>
    /// <returns>The <see cref="LlvmCompatibleTypes"/> representation of the <see cref="Type"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if the <see cref="Type"/> is not recognized.</exception>
    public static LlvmCompatibleTypes ToLlvmType(this Type type)
    {
        if (type.IsByRef || type.IsPointer || type.IsArray || type.IsClass || type.IsInterface || type == typeof(IntPtr) || type == typeof(UIntPtr) || type == typeof(string) || type == typeof(ReadOnlySpan<byte>))
        {
            return LlvmCompatibleTypes.Pointer;
        }

        if (type == typeof(void))
        {
            return LlvmCompatibleTypes.Void;
        }

        if (type.IsValueType)
        {
            if (type == typeof(Half))
            {
                return LlvmCompatibleTypes.Fp16;
            }

            if (type == typeof(BFloat16))
            {
                return LlvmCompatibleTypes.Bf16;
            }

            if (type == typeof(float))
            {
                return LlvmCompatibleTypes.Fp32;
            }

            if (type == typeof(double))
            {
                return LlvmCompatibleTypes.Fp64;
            }

            if (type == typeof(bool))
            {
                return LlvmCompatibleTypes.I1;
            }

            if (type == typeof(byte) || type == typeof(sbyte))
            {
                return LlvmCompatibleTypes.I8;
            }

            if (type == typeof(short) || type == typeof(ushort))
            {
                return LlvmCompatibleTypes.I16;
            }

            if (type == typeof(int) || type == typeof(uint))
            {
                return LlvmCompatibleTypes.I32;
            }

            if (type == typeof(long) || type == typeof(ulong))
            {
                return LlvmCompatibleTypes.I64;
            }
        }

        throw new ArgumentOutOfRangeException(nameof(type), "The type is not recognized.");
    }
}