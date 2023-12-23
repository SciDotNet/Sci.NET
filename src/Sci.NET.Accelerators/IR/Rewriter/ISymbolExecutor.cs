// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.IR.Rewriter.Variables;

namespace Sci.NET.Accelerators.IR.Rewriter;

/// <summary>
/// An interface for a symbol executor.
/// </summary>
[PublicAPI]
public interface ISymbolExecutor
{
    /// <summary>
    /// Executes the symbol.
    /// </summary>
    /// <param name="stack">Gets the current stack.</param>
    /// <param name="instruction">The instruction.</param>
    /// <returns>The result of the execution.</returns>
    public ISsaVariable Execute(ref Stack<ISsaVariable> stack, Instruction<IOperand> instruction);
}