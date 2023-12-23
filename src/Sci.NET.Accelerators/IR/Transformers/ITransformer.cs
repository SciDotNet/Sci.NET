// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Disassembly;
using Sci.NET.Accelerators.IR.Rewriter;

namespace Sci.NET.Accelerators.IR.Transformers;

/// <summary>
/// Represents a transformer.
/// </summary>
[PublicAPI]
public interface ITransformer
{
    /// <summary>
    /// Applies the transformation.
    /// </summary>
    /// <param name="graph">The control flow graph.</param>
    /// <param name="context">The method to transform.</param>
    public void Transform(MsilControlFlowGraph graph, DisassembledMethod context);

    /// <summary>
    /// Reconnects the nodes after a transformation.
    /// </summary>
    /// <param name="graph">The control flow graph.</param>
    /// <param name="removed">The removed nodes.</param>
    /// <param name="replacement">The replacement node.</param>
    public void Reconnect(MsilControlFlowGraph graph, IList<IMsilControlFlowGraphNode> removed, IMsilControlFlowGraphNode replacement);
}