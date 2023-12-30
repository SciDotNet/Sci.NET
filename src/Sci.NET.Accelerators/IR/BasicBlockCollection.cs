// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections;

namespace Sci.NET.Accelerators.IR;

/// <summary>
/// Represents a basic block collection.
/// </summary>
[PublicAPI]
public class BasicBlockCollection : ICollection<BasicBlock>
{
    private readonly ICollection<BasicBlock> _basicBlocks;

    /// <summary>
    /// Initializes a new instance of the <see cref="BasicBlockCollection"/> class.
    /// </summary>
    /// <param name="basicBlocks">The basic blocks.</param>
    public BasicBlockCollection(IEnumerable<BasicBlock> basicBlocks)
    {
        _basicBlocks = basicBlocks.ToList();
    }

    /// <inheritdoc />
    public int Count => _basicBlocks.Count;

    /// <inheritdoc />
    public bool IsReadOnly => _basicBlocks.IsReadOnly;

    /// <inheritdoc/>
    public IEnumerator<BasicBlock> GetEnumerator()
    {
        return _basicBlocks.GetEnumerator();
    }

    /// <inheritdoc/>
    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    /// <inheritdoc />
    public void Add(BasicBlock item)
    {
        _basicBlocks.Add(item);
    }

    /// <inheritdoc />
    public void Clear()
    {
        _basicBlocks.Clear();
    }

    /// <inheritdoc />
    public bool Contains(BasicBlock item)
    {
        return _basicBlocks.Contains(item);
    }

    /// <inheritdoc />
    public void CopyTo(BasicBlock[] array, int arrayIndex)
    {
        _basicBlocks.CopyTo(array, arrayIndex);
    }

    /// <inheritdoc />
    public bool Remove(BasicBlock item)
    {
        return _basicBlocks.Remove(item);
    }
}