// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.NeuralNetworks.Losses;

/// <summary>
/// An interface for a loss function.
/// </summary>
/// <typeparam name="TNumber">The number type of the loss function.</typeparam>
[PublicAPI]
public interface ILossFunction<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Calculates the loss between the predicted and actual values.
    /// </summary>
    /// <param name="y">The actual values.</param>
    /// <param name="yHat">The predicted values.</param>
    /// <returns>The loss between the predicted and actual values.</returns>
    public ITensor<TNumber> CalculateLoss(ITensor<TNumber> y, ITensor<TNumber> yHat);

    /// <summary>
    /// Calculates the gradient of the loss function.
    /// </summary>
    /// <param name="y">The actual values.</param>
    /// <param name="yHat">The predicted values.</param>
    /// <returns>The gradient of the loss function.</returns>
    public ITensor<TNumber> CalculateGradient(ITensor<TNumber> y, ITensor<TNumber> yHat);
}