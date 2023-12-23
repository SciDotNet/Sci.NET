// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Accelerators.IR.Rewriter.Variables;

/// <summary>
///  Represents a constant operand.
/// </summary>
/// <typeparam name="TNumber">The type of the constant.</typeparam>
[PublicAPI]
public interface IConstantSsaVariable<out TNumber> : ISsaVariable
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Gets the operand value.
    /// </summary>
    public TNumber Value { get; }
}