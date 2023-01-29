// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Runtime.CompilerServices;
using Sci.NET.Common.Performance;

namespace Sci.NET.Common.Memory.ReferenceCounting;

/// <summary>
/// A class object which represents a count of references to a block of unmanaged memory.
/// </summary>
[PublicAPI]
[DebuggerDisplay("Count: {_count}, Disposed: {_isDisposed}")]
public sealed record ReferenceCount : IDisposable
{
    private readonly Guid _id;
    private long _count;
    private bool _isDisposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="ReferenceCount"/> class.
    /// </summary>
    public ReferenceCount()
    {
        _id = Guid.NewGuid();
        _count = 0;
    }

    /// <summary>
    /// Gets the number of references to the instance.
    /// </summary>
    /// <returns>The number of references to the instance.</returns>
    /// <remarks>Cannot use a property since the internal count
    /// is passed by ref.</remarks>
#pragma warning disable CA1024
    public long GetCount()
#pragma warning restore CA1024
    {
        return _count;
    }

    /// <summary>
    /// Increments the reference count.
    /// </summary>
    [MethodImpl(ImplementationOptions.HotPath)]
    public void Increment()
    {
        AssertNotDisposed();
        _ = Interlocked.Increment(ref _count);
    }

    /// <summary>
    /// Decrements the reference count.
    /// </summary>
    [MethodImpl(ImplementationOptions.HotPath)]
    public void Decrement()
    {
        AssertNotDisposed();
        if (Interlocked.Read(ref _count) != 0)
        {
            _ = Interlocked.Decrement(ref _count);
        }
    }

    /// <inheritdoc />
    public bool Equals(ReferenceCount? other)
    {
        if (other is null)
        {
            return false;
        }

        AssertNotDisposed();
        other.AssertNotDisposed();

        return _id == other._id && CountEquals(other);
    }

    /// <summary>
    /// Determines if the reference count is equal to another reference count.
    /// </summary>
    /// <param name="other">The reference count to compare to.</param>
    /// <returns><c>true</c> if the counts are equal, else <c>false</c>.</returns>
    [MethodImpl(ImplementationOptions.HotPath)]
    public bool CountEquals(ReferenceCount other)
    {
        var xCount = Interlocked.Read(ref _count);
        var yCount = Interlocked.Read(ref other._count);

        return xCount == yCount;
    }

    /// <inheritdoc />
    public override int GetHashCode()
    {
        AssertNotDisposed();
        return _id.GetHashCode();
    }

    /// <summary>
    /// Determines if the reference count is zero.
    /// </summary>
    /// <returns><c>true</c> if the count is zero, else <c>false</c>.</returns>
    [MethodImpl(ImplementationOptions.HotPath)]
    public bool IsZero()
    {
        AssertNotDisposed();
        return Interlocked.Read(ref _count) == 0;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _isDisposed = true;
    }

    [MethodImpl(ImplementationOptions.HotPath)]
    private void AssertNotDisposed()
    {
        if (!_isDisposed)
        {
            return;
        }

        throw new ObjectDisposedException(nameof(ReferenceCount), "The reference count has been disposed.");
    }
}