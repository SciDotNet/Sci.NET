// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Sci.NET.Accelerators.IR;
using Sci.NET.Accelerators.Rewriter.Variables;

namespace Sci.NET.Accelerators.Rewriter;

/// <summary>
/// Represents a method in MSIL (Microsoft Intermediate Language) with Static Single Assignment (SSA) form.
/// </summary>
[SuppressMessage("StyleCop.CSharp.OrderingRules", "SA1206:Declaration keywords should follow order", Justification = "StyleCop does not support required members.")]
public class MsilSsaMethod
{
    /// <summary>
    /// Gets the collection of basic blocks.
    /// </summary>
    public required ICollection<BasicBlock> BasicBlocks { get; init; }

    /// <summary>
    /// Gets the collection of parameters.
    /// </summary>
    public required ICollection<ParameterSsaVariable> Parameters { get; init; }

    /// <summary>
    /// Gets the collection of local variables.
    /// </summary>
    public required ICollection<LocalVariableSsaVariable> Locals { get; init; }

    /// <summary>
    /// Gets the return type.
    /// </summary>
    public required Type ReturnType { get; init; }
}