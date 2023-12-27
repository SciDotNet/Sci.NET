// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.Disassembly.Cfg;
using Sci.NET.Accelerators.IR.Rewriter;

namespace Sci.NET.Accelerators.IR.Builders;

/// <summary>
/// Represents an IR builder.
/// </summary>
[PublicAPI]
public static class IrBuilder
{
    /// <summary>
    /// Builds the IR from the disassembled method.
    /// </summary>
    /// <param name="method">The disassembled method.</param>
    /// <returns>The IR.</returns>
    public static IReadOnlyList<BasicBlock> Build(DisassembledMethod method)
    {
        var cfg = MsilControlFlowGraph.Create(method.Instructions);
        var executor = new SymbolicExecutor(cfg, method);
        var ssaMethod = executor.Execute();
        var basicBlocks = BasicBlockBuilder.CreateBasicBlocks(cfg, ssaMethod.Instructions);

        LoopDetector.Detect(new BasicBlockCollection(basicBlocks));

        return basicBlocks;
    }
}