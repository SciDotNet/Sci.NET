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

    public ITensor<TNumber> LeakyReLU<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.LeakyReLU(value, result, alpha);

        return result;
    }

    public ITensor<TNumber> LeakyReLUPrime<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.LeakyReLUPrime(value, result, alpha);

        return result;
    }

    public ITensor<TNumber> Elu<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.Elu(value, result, alpha);

        return result;
    }

    public ITensor<TNumber> EluPrime<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.EluPrime(value, result, alpha);

        return result;
    }

    public ITensor<TNumber> Celu<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.Celu(value, result, alpha);

        return result;
    }

    public ITensor<TNumber> CeluPrime<TNumber>(ITensor<TNumber> value, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.CeluPrime(value, result, alpha);

        return result;
    }

    public ITensor<TNumber> Swish<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.Swish(value, result);

        return result;
    }

    public ITensor<TNumber> SwishPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.SwishPrime(value, result);

        return result;
    }

    public ITensor<TNumber> Mish<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IHyperbolicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.Mish(value, result);

        return result;
    }

    public ITensor<TNumber> MishPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IHyperbolicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.MishPrime(value, result);

        return result;
    }

    public ITensor<TNumber> HardTanh<TNumber>(ITensor<TNumber> value, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.HardTanh(
            value,
            result,
            min,
            max);

        return result;
    }

    public ITensor<TNumber> HardTanhPrime<TNumber>(ITensor<TNumber> value, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.HardTanhPrime(
            value,
            result,
            min,
            max);

        return result;
    }

    public ITensor<TNumber> HardSigmoid<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.HardSigmoid(value, result);

        return result;
    }

    public ITensor<TNumber> HardSigmoidPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.HardSigmoidPrime(value, result);

        return result;
    }

    public ITensor<TNumber> LogSigmoid<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.LogSigmoid(value, result);

        return result;
    }

    public ITensor<TNumber> LogSigmoidPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.LogSigmoidPrime(value, result);

        return result;
    }

    public ITensor<TNumber> GELU<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.GELU(value, result);

        return result;
    }

    public ITensor<TNumber> GELUPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.GELUPrime(value, result);

        return result;
    }

    public ITensor<TNumber> SoftPlus<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.SoftPlus(value, result);

        return result;
    }

    public ITensor<TNumber> SoftPlusPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.SoftPlusPrime(value, result);

        return result;
    }

    public ITensor<TNumber> SoftSign<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.SoftSign(value, result);

        return result;
    }

    public ITensor<TNumber> SoftSignPrime<TNumber>(ITensor<TNumber> value)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(value.Shape, value.Backend);

        value.Backend.ActivationFunctions.SoftSignPrime(value, result);

        return result;
    }
}