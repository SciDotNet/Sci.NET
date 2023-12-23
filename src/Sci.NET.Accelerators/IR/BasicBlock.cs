// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// Represents a basic block.
/// </summary>
[PublicAPI]
public sealed class BasicBlock : IEquatable<BasicBlock>
{
    private readonly Guid _blockId = Guid.NewGuid();

    /// <summary>
    /// Gets the instructions of the basic block.
    /// </summary>
    public IList<IControlFlowGraphNode> Instructions { get; init; } = new List<IControlFlowGraphNode>();

    /// <summary>
    /// Gets the next basic blocks which can be reached from this basic block.
    /// </summary>
    public IList<BasicBlock> NextBlocks { get; init; } = new List<BasicBlock>();

    /// <summary>
    /// Gets the start offset of the basic block.
    /// </summary>
    public int StartOffset { get; init; }

    /// <summary>
    /// Gets the end offset of the basic block.
    /// </summary>
    public int EndOffset { get; internal set; }

    /// <summary>
    /// Gets a value indicating whether the basic block is a loop.
    /// </summary>
    public bool IsLoop { get; internal set; }

    /// <inheritdoc />
    public bool Equals(BasicBlock? other)
    {
        if (other is null)
        {
            return false;
        }

        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return _blockId.Equals(other._blockId);
    }

    /// <inheritdoc />
    public override bool Equals(object? obj)
    {
        if (obj is null)
        {
            return false;
        }

        if (ReferenceEquals(this, obj))
        {
            return true;
        }

        if (obj.GetType() != GetType())
        {
            return false;
        }

        return Equals((BasicBlock)obj);
    }

    /// <inheritdoc />
    public override int GetHashCode()
    {
        return HashCode.Combine(_blockId);
    }
}