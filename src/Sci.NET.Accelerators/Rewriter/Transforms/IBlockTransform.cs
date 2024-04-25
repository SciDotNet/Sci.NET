// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.IR;

namespace Sci.NET.Accelerators.Rewriter.Transforms;

/// <summary>
/// Represents a block transform.
/// </summary>
[PublicAPI]
public interface IBlockTransform
{
    /// <summary>
    /// Transforms the specified blocks.
    /// </summary>
    /// <param name="allBlocks">The blocks to transform.</param>
    public void Transform(ICollection<BasicBlock> allBlocks);
}