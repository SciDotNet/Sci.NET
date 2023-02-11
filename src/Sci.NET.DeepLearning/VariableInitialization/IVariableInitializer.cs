// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.DeepLearning.VariableInitialization;

/// <summary>
/// An interface for variable initializers.
/// </summary>
/// <typeparam name="TNumber">The number type to create.</typeparam>
[PublicAPI]
public interface IVariableInitializer<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Gets the minimum value to be generated.
    /// </summary>
    public TNumber Min { get; }

    /// <summary>
    /// Gets the maximum value to be generated.
    /// </summary>
    public TNumber Max { get; }

    /// <summary>
    /// Initializes a new <see cref="ITensor{TNumber}"/> with the specified shape.
    /// </summary>
    /// <param name="shape">The shape of the <see cref="ITensor{TNumber}"/> to create.</param>
    /// <returns>A new randomly initialized <see cref="ITensor{TNumber}"/> within the constraints of the <see cref="IVariableInitializer{TNumber}"/>.</returns>
    public ITensor<TNumber> Initialize(Shape shape);
}