// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Accelerators.Disassembly.Operands;

/// <summary>
/// Represents an MSIL operand with a number value.
/// </summary>
/// <typeparam name="TNumber">The number type.</typeparam>
[PublicAPI]
public interface IMsilInlineNumberOperand<out TNumber> : IMsilOperand
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Gets the operand value.
    /// </summary>
    public TNumber Value { get; }
}