// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace Sci.NET.BLAS.Types;

/// <summary>
/// Enumerates the number types supported by the
/// Sci.NET Native BLAS libraries.
/// </summary>
[PublicAPI]
[SuppressMessage(
    "Naming",
    "CA1720:Identifier contains type name",
    Justification = "This is the correct name for this enum.")]
public enum NumberType
{
    /// <summary>
    /// Unknown number type.
    /// </summary>
    Unknown = 0,

    /// <summary>
    /// Single precision floating point numbers.
    /// </summary>
    Float32 = 1,

    /// <summary>
    /// Double precision floating point numbers.
    /// </summary>
    Float64 = 2
}