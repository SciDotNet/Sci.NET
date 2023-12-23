// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Disassembly;

namespace Sci.NET.Accelerators.IR;

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
    public void Transform(ControlFlowGraph graph, DisassembledMethod context);

    /// <summary>
    /// Reconnects the nodes after a transformation.
    /// </summary>
    /// <param name="graph">The control flow graph.</param>
    /// <param name="removed">The removed nodes.</param>
    /// <param name="replacement">The replacement node.</param>
    public void Reconnect(ControlFlowGraph graph, IList<IControlFlowGraphNode> removed, IControlFlowGraphNode replacement);
}