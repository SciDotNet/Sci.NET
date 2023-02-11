// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.DeepLearning.VariableInitialization;

/// <summary>
/// A class for variable initializers.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
[PublicAPI]
public class VariableInitializer<TNumber> : IVariableInitializer<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="VariableInitializer{TNumber}"/> class.
    /// </summary>
    /// <param name="min">The minimum number to generate.</param>
    /// <param name="max">The maximum number to generate.</param>
    /// <param name="seed">The seed for the random number generator.</param>
    public VariableInitializer(TNumber min, TNumber max, long seed)
    {
        Min = min;
        Max = max;
        Seed = seed;
    }

    /// <inheritdoc />
    public TNumber Min { get; }

    /// <inheritdoc />
    public TNumber Max { get; }

    /// <summary>
    /// Gets or sets the seed for the random number generator.
    /// </summary>
    public long Seed { get; set; }

    /// <inheritdoc />
    public ITensor<TNumber> Initialize(Shape shape)
    {
        return Tensor.Random.Uniform(shape, Min, Max, Seed);
    }
}