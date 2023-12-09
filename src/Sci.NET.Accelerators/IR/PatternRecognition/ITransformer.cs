// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Disassembly;

namespace Sci.NET.Accelerators.IR.PatternRecognition;

/// <summary>
/// Represents a transformer.
/// </summary>
[PublicAPI]
public interface ITransformer
{
    /// <summary>
    /// Applies the transformation.
    /// </summary>
    /// <param name="method">The method to transform.</param>
    public void Transform(DisassembledMethod method);
}