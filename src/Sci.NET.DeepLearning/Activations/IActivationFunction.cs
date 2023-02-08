// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.DeepLearning.Layers;

namespace Sci.NET.DeepLearning.Activations;

/// <summary>
/// An interface for activation functions.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="IActivationFunction{TNumber}"/>.</typeparam>
[PublicAPI]
public interface IActivationFunction<TNumber> : ILayer<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
}