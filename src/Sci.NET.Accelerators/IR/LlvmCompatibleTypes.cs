// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// Represents the possible LLVM types.
/// </summary>
[PublicAPI]
[SuppressMessage("Naming", "CA1720:Identifier contains type name", Justification = "LLVM is a type name.")]
public enum LlvmCompatibleTypes
{
    /// <summary>
    /// The void type.
    /// </summary>
    Void = 0,

    /// <summary>
    /// The half precision floating point type.
    /// </summary>
    Fp16 = 1,

    /// <summary>
    /// The half precision floating point type.
    /// </summary>
    Bf16 = 2,

    /// <summary>
    /// The single precision floating point type.
    /// </summary>
    Fp32 = 3,

    /// <summary>
    /// The double precision floating point type.
    /// </summary>
    Fp64 = 4,

    /// <summary>
    /// The 1-bit integer type.
    /// </summary>
    I1 = 5,

    /// <summary>
    /// The 8-bit signed integer type.
    /// </summary>
    I8 = 6,

    /// <summary>
    /// The 16-bit signed integer type.
    /// </summary>
    I16 = 7,

    /// <summary>
    /// The 32-bit signed integer type.
    /// </summary>
    I32 = 8,

    /// <summary>
    /// The 64-bit signed integer type.
    /// </summary>
    I64 = 9,

    /// <summary>
    /// The 128-bit signed integer type.
    /// </summary>
    I128 = 10,

    /// <summary>
    /// The pointer type.
    /// </summary>
    Pointer = 11,
}