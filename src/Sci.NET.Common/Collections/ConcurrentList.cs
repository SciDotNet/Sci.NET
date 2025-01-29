// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace Sci.NET.Common.Collections;

/// <summary>
/// Represents a thread-safe list.
/// </summary>
/// <typeparam name="T">The type of the elements in the list.</typeparam>
[PublicAPI]
public class ConcurrentList<T> : IDisposable
{
    private readonly ReaderWriterLockSlim _lock;
    private readonly List<T> _list;

    /// <summary>
    /// Initializes a new instance of the <see cref="ConcurrentList{T}"/> class.
    /// </summary>
    /// <param name="capacity">The initial capacity of the list.</param>
    [ExcludeFromCodeCoverage]
    public ConcurrentList(int capacity = 0)
    {
        _lock = new ReaderWriterLockSlim();
        _list = new List<T>(capacity);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ConcurrentList{T}"/> class.
    /// </summary>
    /// <param name="readerWriterLock">The reader-writer lock to use for thread safety.</param>
    /// <param name="capacity">The initial capacity of the list.</param>
    public ConcurrentList(ReaderWriterLockSlim readerWriterLock, int capacity = 0)
    {
        _lock = readerWriterLock;
        _list = new List<T>(capacity);
    }

    /// <summary>
    /// Gets the number of elements in the list.
    /// </summary>
    public int Count => GetCount();

    /// <summary>
    /// Gets or sets the element at the specified index.
    /// </summary>
    /// <param name="index">The index of the element to get or set.</param>
    public T this[int index]
    {
        get => ElementAt(index);
        set => ReplaceAt(index, value);
    }

    /// <summary>
    /// Adds an element to the list.
    /// </summary>
    /// <param name="item">The element to add.</param>
    public void Add(T item)
    {
        _lock.EnterWriteLock();

        try
        {
            _list.Add(item);
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Adds a range of elements to the list.
    /// </summary>
    /// <param name="items">The elements to add.</param>
    public void AddRange(IEnumerable<T> items)
    {
        _lock.EnterWriteLock();

        try
        {
            _list.AddRange(items);
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Inserts an element into the list at the specified index.
    /// </summary>
    /// <param name="index">The index at which to insert the element.</param>
    /// <param name="item">The element to insert.</param>
    public void Insert(int index, T item)
    {
        _lock.EnterWriteLock();

        try
        {
            _list.Insert(index, item);
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Gets the index of the specified element.
    /// </summary>
    /// <param name="index">The index of the element to get.</param>
    /// <returns>The element at the specified index.</returns>
    public T ElementAt(int index)
    {
        _lock.EnterReadLock();

        try
        {
            return _list[index];
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>
    /// Replace the element at the specified index.
    /// </summary>
    /// <param name="index">The index of the element to replace.</param>
    /// <param name="item">The element to replace.</param>
    public void ReplaceAt(int index, T item)
    {
        _lock.EnterWriteLock();

        try
        {
            _list[index] = item;
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Determines whether the list contains the specified element.
    /// </summary>
    /// <param name="item">The element to check for.</param>
    /// <returns>True if the list contains the element, false otherwise.</returns>
    public bool Contains(T item)
    {
        _lock.EnterReadLock();

        try
        {
            return _list.Contains(item);
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>
    /// Removes the specified element from the list.
    /// </summary>
    /// <param name="element">The element to remove.</param>
    /// <returns>True if the element was removed, false otherwise.</returns>
    public bool Remove(T element)
    {
        _lock.EnterWriteLock();

        try
        {
            return _list.Remove(element);
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes of the resources used by the <see cref="ConcurrentList{T}"/>.
    /// </summary>
    /// <param name="disposing">True if disposing, false otherwise.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            _lock.Dispose();
        }
    }

    private int GetCount()
    {
        _lock.EnterReadLock();

        try
        {
            return _list.Count;
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }
}