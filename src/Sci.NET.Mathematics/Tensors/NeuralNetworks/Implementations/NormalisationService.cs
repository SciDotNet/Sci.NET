// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Numerics;
using Sci.NET.Mathematics.Tensors.Common;

namespace Sci.NET.Mathematics.Tensors.NeuralNetworks.Implementations;

internal class NormalisationService : INormalisationService
{
    private readonly IGradientAppenderService _gradientAppenderService;

    public NormalisationService()
    {
        _gradientAppenderService = TensorServiceProvider.GetTensorOperationServiceProvider().GetGradientAppenderService();
    }

    public Matrix<TNumber> BatchNorm1dForward<TNumber>(Matrix<TNumber> input, Vector<TNumber> scale, Vector<TNumber> bias)
        where TNumber : unmanaged, INumber<TNumber>, IRootFunctions<TNumber>
    {
        var mean = input.Mean(new int[] { 0 }).ToVector();

        using var m = new Scalar<TNumber>(TNumber.CreateChecked(mean.Shape[0]));
        m.To(input.Device);

        var variance = input.Variance(0).ToVector();
        using var epsilon = new Scalar<TNumber>(GenericMath.ScaledEpsilon<TNumber>(10000));
        using var epsilonRoot = variance.Add(epsilon).Sqrt();
        using var difference = input.Subtract(mean);
        using var norm = difference.Divide(epsilonRoot);
        var scaleNorm = norm.Multiply(scale);

        return scaleNorm.Add(bias);
    }

    public ITensor<TNumber> Clip<TNumber>(ITensor<TNumber> tensor, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);

        result.Backend.Normalisation.Clip(
            tensor,
            result,
            min,
            max);

        _gradientAppenderService.AddGradientIfRequired(
            ref result,
            tensor,
            null,
            grad => grad.Multiply(ClipBackward(tensor, min, max)));

        return result;
    }

    private static Tensor<TNumber> ClipBackward<TNumber>(ITensor<TNumber> tensor, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.Shape, tensor.Backend);

        result.Backend.Normalisation.ClipBackward(
            tensor,
            result,
            min,
            max);

        return result;
    }
}