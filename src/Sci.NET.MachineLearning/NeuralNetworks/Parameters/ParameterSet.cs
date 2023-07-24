// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Numerics;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.NeuralNetworks.Parameters;

/// <summary>
/// A group of parameters.
/// </summary>
/// <typeparam name="TNumber">The number type of the parameters.</typeparam>
[PublicAPI]
public class ParameterSet<TNumber> : ICollection<NamedParameter<TNumber>>, ITensorLocalityOperations, IDisposable
    where TNumber : unmanaged, INumber<TNumber>
{
    private readonly List<NamedParameter<TNumber>> _namedParameters;

    /// <summary>
    /// Initializes a new instance of the <see cref="ParameterSet{TNumber}"/> class.
    /// </summary>
    public ParameterSet()
    {
        _namedParameters = new List<NamedParameter<TNumber>>();
    }

    /// <inheritdoc />
    public int Count => _namedParameters.Count;

    /// <inheritdoc />
    public bool IsReadOnly => false;

    /// <summary>
    /// Gets the <see cref="NamedParameter{TNumber}"/> with the specified name.
    /// </summary>
    /// <param name="name">The name of the parameter.</param>
    /// <exception cref="ArgumentException">The parameter with the given name does not exist in the group.</exception>
    public NamedParameter<TNumber> this[string name] =>
        _namedParameters.Find(p => p.Name == name) ?? throw new ArgumentException($"Parameter {name} not found.");

    /// <inheritdoc />
    public void Add(NamedParameter<TNumber> item)
    {
        _namedParameters.Add(item);
    }

    /// <summary>
    /// Adds a <see cref="NamedParameter{TNumber}"/> to the group.
    /// </summary>
    /// <param name="name">The name of the parameter.</param>
    /// <param name="shape">The shape of the parameter.</param>
    public void Add(string name, Shape shape)
    {
        _namedParameters.Add(new NamedParameter<TNumber>(name, shape));
    }

    /// <inheritdoc />
    public void Clear()
    {
        _namedParameters.Clear();
    }

    /// <inheritdoc />
    public bool Contains(NamedParameter<TNumber> item)
    {
        return _namedParameters.Contains(item);
    }

    /// <inheritdoc />
    public void CopyTo(NamedParameter<TNumber>[] array, int arrayIndex)
    {
        _namedParameters.CopyTo(array, arrayIndex);
    }

    /// <inheritdoc />
    public bool Remove(NamedParameter<TNumber> item)
    {
        return _namedParameters.Remove(item);
    }

    /// <inheritdoc />
    public void To<TDevice>()
        where TDevice : IDevice, new()
    {
        _namedParameters.ForEach(x => x.To<TDevice>());
    }

    /// <inheritdoc />
    public IEnumerator<NamedParameter<TNumber>> GetEnumerator()
    {
        return _namedParameters.GetEnumerator();
    }

    /// <inheritdoc/>
    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes of the <see cref="ParameterSet{TNumber}"/>.
    /// </summary>
    /// <param name="disposing">Whether the <see cref="ParameterSet{TNumber}"/> is disposing.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            foreach (var namedParameter in _namedParameters)
            {
                namedParameter.Dispose();
            }
        }
    }
}