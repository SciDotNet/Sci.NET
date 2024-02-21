// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Numerics;

namespace Sci.NET.Mathematics.Tensors.NeuralNetworks.Implementations;

internal class NormalisationService : INormalisationService
{
    public Matrix<TNumber> BatchNorm1dForward<TNumber>(Matrix<TNumber> input, Vector<TNumber> scale, Vector<TNumber> bias)
        where TNumber : unmanaged, INumber<TNumber>, IRootFunctions<TNumber>
    {
        var mean = input.Mean(new int[] { 1 }).ToVector();

        using var m = new Scalar<TNumber>(TNumber.CreateChecked(mean.Shape[0]));
        m.To(input.Device);

        var variance = mean.Subtract(input).Square().Sum(new int[] { 1 }).Divide(m).ToVector();
        using var epsilon = new Scalar<TNumber>(GenericMath.Epsilon<TNumber>() * TNumber.CreateChecked(1000));
        var result = new Matrix<TNumber>(input.Rows, input.Columns);

        result.To(input.Device);
        epsilon.To(input.Device);

        input.Backend.NeuralNetworks.BatchNorm1dForward(
            input,
            scale,
            bias,
            mean,
            variance,
            result,
            epsilon);

        return result;
    }
}