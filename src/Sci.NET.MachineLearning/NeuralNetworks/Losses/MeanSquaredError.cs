// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.NeuralNetworks.Losses;

/// <summary>
/// A mean squared error loss function.
/// </summary>
/// <typeparam name="TNumber">The number type of the loss function.</typeparam>
[PublicAPI]
public class MeanSquaredError<TNumber> : ILossFunction<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <inheritdoc />
    public ITensor<TNumber> CalculateLoss(ITensor<TNumber> y, ITensor<TNumber> yHat)
    {
        var backend = y.Backend;

        using var m = new Scalar<TNumber>(TNumber.CreateChecked(y.Shape.Dimensions[0]), backend);
        using var error = y - yHat;
        using var squaredError = error.Square();
        using var sum = squaredError.Sum(new int[] { 1 });
        return sum / m;
    }

    /// <inheritdoc />
    public ITensor<TNumber> CalculateGradient(ITensor<TNumber> y, ITensor<TNumber> yHat)
    {
        using var minusTwo = new Scalar<TNumber>(TNumber.CreateChecked(-2), y.Backend);
        using var m = new Scalar<TNumber>(TNumber.CreateChecked(y.Shape.Dimensions[0]), y.Backend);
        return minusTwo * (y - yHat) / m;
    }
}