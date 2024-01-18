// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Sci.NET.Accelerators.Disassembly.Operands;

namespace Sci.NET.Accelerators.Disassembly;

/// <summary>
/// Represents a disassembled MSIL method.
/// </summary>
[PublicAPI]
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members yet.")]
public class DisassembledMsilMethod
{
    /// <summary>
    /// Gets the MSIL method metadata.
    /// </summary>
    public required MsilMethodMetadata Metadata { get; init; }

    /// <summary>
    /// Gets the instructions.
    /// </summary>
    public required IReadOnlyCollection<MsilInstruction<IMsilOperand>> Instructions { get; init; }
}