// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.NeuralNetworks.Implementations;

internal class ActivationFunctionService : IActivationFunctionService
{
    public ITensor<TNumber> Sigmoid<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);
        value.Backend.ActivationFunctions.Sigmoid(value, result);

        return result;
    }

    public ITensor<TNumber> SigmoidPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);
        value.Backend.ActivationFunctions.SigmoidPrime(value, result);

        return result;
    }

    public ITensor<TNumber> ReLU<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);
        value.Backend.ActivationFunctions.ReLU(value, result);

        return result;
    }

    public ITensor<TNumber> ReLUPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);
        value.Backend.ActivationFunctions.ReLUPrime(value, result);

        return result;
    }

    public ITensor<TNumber> Softmax<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        using var expScores = value.Exp();
        using var sumExpScores = expScores.Sum();
        return expScores.Divide(sumExpScores);
    }

    public ITensor<TNumber> SoftmaxPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        using var one = new Scalar<TNumber>(TNumber.CreateChecked(1));
        using var softmax = value.Softmax().ToTensor();
        return softmax.Multiply(one.Subtract(softmax));
    }
}