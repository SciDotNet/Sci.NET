// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Numerics;

namespace Sci.NET.MachineLearning.NeuralNetworks.Parameters;

/// <summary>
/// A collection of <see cref="ParameterSet{TNumber}"/>s.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="ParameterCollection{TNumber}"/>.</typeparam>
[PublicAPI]
public class ParameterCollection<TNumber> : ICollection<ParameterSet<TNumber>>
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly List<ParameterSet<TNumber>> _parameterSets;

    /// <summary>
    /// Initializes a new instance of the <see cref="ParameterCollection{TNumber}"/> class.
    /// </summary>
    public ParameterCollection()
    {
        _parameterSets = new List<ParameterSet<TNumber>>();
    }

    /// <inheritdoc />
    public int Count { get; }

    /// <inheritdoc />
    public bool IsReadOnly { get; }

    /// <summary>
    /// Gets the <see cref="ParameterSet{TNumber}"/> at the specified index.
    /// </summary>
    /// <param name="i">The index of the <see cref="ParameterSet{TNumber}"/> to get.</param>
    public ParameterSet<TNumber> this[int i] => _parameterSets[i];

    /// <inheritdoc />
    public void Add(ParameterSet<TNumber> item)
    {
        _parameterSets.Add(item);
    }

    /// <inheritdoc />
    public void Clear()
    {
        _parameterSets.Clear();
    }

    /// <inheritdoc />
    public bool Contains(ParameterSet<TNumber> item)
    {
        return _parameterSets.Contains(item);
    }

    /// <inheritdoc />
    public void CopyTo(ParameterSet<TNumber>[] array, int arrayIndex)
    {
        _parameterSets.CopyTo(array, arrayIndex);
    }

    /// <inheritdoc />
    public bool Remove(ParameterSet<TNumber> item)
    {
        return _parameterSets.Remove(item);
    }

    /// <summary>
    /// Gets all the parameters in the collection.
    /// </summary>
    /// <returns>An enumerable of all the parameters in the collection.</returns>
    public IEnumerable<NamedParameter<TNumber>> GetAll()
    {
        var parameters = new List<NamedParameter<TNumber>>();

        foreach (var collection in _parameterSets)
        {
            parameters.AddRange(collection);
        }

        return parameters;
    }

    /// <inheritdoc />
    public IEnumerator<ParameterSet<TNumber>> GetEnumerator()
    {
        return _parameterSets.GetEnumerator();
    }

    /// <inheritdoc/>
    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}