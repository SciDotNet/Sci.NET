// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedActivationFunctionKernels : IActivationFunctionKernels
{
    public void Sigmoid<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => outputMemory[i] = TNumber.One / (TNumber.One + TNumber.Exp(-inputMemory[i])));
    }

    public void SigmoidPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var sigmoid = TNumber.One / (TNumber.One + TNumber.Exp(-inputMemory[i]));
                outputMemory[i] = sigmoid * (TNumber.One - sigmoid);
            });
    }

    public void ReLU<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => outputMemory[i] = inputMemory[i] > TNumber.Zero ? inputMemory[i] : TNumber.Zero);
    }

    public void ReLUPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => outputMemory[i] = inputMemory[i] > TNumber.Zero ? TNumber.One : TNumber.Zero);
    }

    public void LeakyReLU<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => outputMemory[i] = inputMemory[i] > TNumber.Zero ? inputMemory[i] : alpha * inputMemory[i]);
    }

    public void LeakyReLUPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => outputMemory[i] = inputMemory[i] > TNumber.Zero ? TNumber.One : alpha);
    }

    public void Elu<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => outputMemory[i] = inputMemory[i] > TNumber.Zero ? inputMemory[i] : alpha * (TNumber.Exp(inputMemory[i]) - TNumber.One));
    }

    public void EluPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => outputMemory[i] = inputMemory[i] > TNumber.Zero ? TNumber.One : alpha * TNumber.Exp(inputMemory[i]));
    }

    public void Celu<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => outputMemory[i] = TNumber.Max(TNumber.Zero, inputMemory[i]) + TNumber.Min(TNumber.Zero, alpha * (TNumber.Exp(inputMemory[i] / alpha) - TNumber.One)));
    }

    public void CeluPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber alpha)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => outputMemory[i] = TNumber.Min(TNumber.One, alpha * TNumber.Exp(inputMemory[i] / alpha)));
    }

    public void Swish<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => outputMemory[i] = inputMemory[i] * (TNumber.One / (TNumber.One + TNumber.Exp(-inputMemory[i]))));
    }

    public void SwishPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var expNegX = TNumber.Exp(-inputMemory[i]);
                outputMemory[i] = ((expNegX * (inputMemory[i] + TNumber.One)) + TNumber.One) / ((TNumber.One + expNegX) * (TNumber.One + expNegX));
            });
    }

    public void Mish<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IHyperbolicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                // mish\left(x\right)=x\tanh\left(\ln\left(1+e^{x}\right)\right)
                // mish\left(x\right)=x\cdot\frac{e^{\ln\left(e^{x}+1\right)}-e^{-\ln\left(e^{x}+1\right)}}{e^{\ln\left(e^{x}+1\right)}+e^{-\ln\left(e^{x}+1\right)}}
                var expXPlus1 = TNumber.Exp(inputMemory[i]) + TNumber.One;
                var lnExpXPlus1 = TNumber.Log(expXPlus1);
                var expLnExpPlus1 = TNumber.Exp(lnExpXPlus1);
                var negExpLnExpPlus1 = TNumber.Exp(-lnExpXPlus1);

                outputMemory[i] = inputMemory[i] * (expLnExpPlus1 - negExpLnExpPlus1) / (expLnExpPlus1 + negExpLnExpPlus1);
            });
    }

    public void MishPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IHyperbolicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;
        var two = TNumber.CreateChecked(2);
        var minusOne = TNumber.Zero - TNumber.One;
        var minusTwo = TNumber.Zero - two;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                // mish'\left(x\right)=\frac{-1+\left(1+e^{x}\right)^{2}}{1+\left(1+e^{x}\right)^{2}}-\frac{2e^{x}\left(1+e^{x}\right)\left(-1+\left(1+e^{x}\right)^{2}\right)x}{\left(1+\left(1+e^{x}\right)^{2}\right)^{2}}+\frac{2e^{x}\left(1+e^{x}\right)x}{1+\left(1+e^{x}\right)^{2}}
                var eToTheX = TNumber.Exp(inputMemory[i]);
                var onePlusEX = TNumber.One + eToTheX;
                var onePlusEXSquared = onePlusEX * onePlusEX;

                // \frac{-1+\left(1+e^{x}\right)^{2}}{1+\left(1+e^{x}\right)^{2}}
                var firstTerm = (minusOne + onePlusEXSquared) / (TNumber.One + onePlusEXSquared);

                // \frac{2e^{x}\left(1+e^{x}\right)\left(-1+\left(1+e^{x}\right)^{2}\right)x}{\left(1+\left(1+e^{x}\right)^{2}\right)^{2}}
                var onePlusExpXSquared = (TNumber.One + onePlusEXSquared) * (TNumber.One + onePlusEXSquared);
                var secondTerm = minusTwo * eToTheX * onePlusEX * (minusOne + onePlusEXSquared) * inputMemory[i] / onePlusExpXSquared;

                // \frac{2e^{x}\left(1+e^{x}\right)x}{1+\left(1+e^{x}\right)^{2}}
                var thirdTerm = two * eToTheX * onePlusEX * inputMemory[i] / (TNumber.One + onePlusEXSquared);

                outputMemory[i] = firstTerm + secondTerm + thirdTerm;
            });
    }

    public void HardTanh<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
#pragma warning disable IDE0045
                if (inputMemory[i] < min)
                {
                    outputMemory[i] = min;
                }
                else if (inputMemory[i] > max)
                {
                    outputMemory[i] = max;
                }
                else
                {
                    outputMemory[i] = inputMemory[i];
                }
#pragma warning restore IDE0045
            });
    }

    public void HardTanhPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
#pragma warning disable IDE0045
                if (inputMemory[i] <= min || inputMemory[i] >= max)
                {
                    outputMemory[i] = TNumber.Zero;
                }
                else
                {
                    outputMemory[i] = TNumber.One;
                }
#pragma warning restore IDE0045
            });
    }

    public void HardSigmoid<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;
        var minusOne = TNumber.Zero - TNumber.One;
        var two = TNumber.CreateChecked(2);

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
#pragma warning disable IDE0045
                if (inputMemory[i] <= minusOne)
                {
                    outputMemory[i] = TNumber.Zero;
                }
                else if (inputMemory[i] >= TNumber.One)
                {
                    outputMemory[i] = TNumber.One;
                }
                else
                {
                    outputMemory[i] = inputMemory[i] / two;
                }
#pragma warning restore IDE0045
            });
    }

    public void HardSigmoidPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;
        var two = TNumber.CreateChecked(2);

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                outputMemory[i] = inputMemory[i] <= -TNumber.One || inputMemory[i] >= TNumber.One
                    ? TNumber.Zero
                    : TNumber.One / two;
            });
    }

    public void LogSigmoid<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => outputMemory[i] = -TNumber.Log(TNumber.One + TNumber.Exp(-inputMemory[i])));
    }

    public void LogSigmoidPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => outputMemory[i] = TNumber.One / (TNumber.One + TNumber.Exp(inputMemory[i])));
    }

    public void GELU<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;
        var two = TNumber.CreateChecked(2);
        var magicNumber = TNumber.CreateChecked(0.044715);
        var sqrtPiTerm = TNumber.Sqrt(two / TNumber.Pi);

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var x = inputMemory[i];
                var halfX = x / two;
                var xCubed = x * x * x;
                var tanhArg = sqrtPiTerm * (x + (magicNumber * xCubed));

                outputMemory[i] = halfX * (TNumber.One + TNumber.Tanh(tanhArg));
            });
    }

    public void GELUPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;
        var magicNumber1 = TNumber.CreateChecked(0.3989422804014327);
        var magicNumber2 = TNumber.CreateChecked(0.134145);
        var magicNumber3 = TNumber.CreateChecked(0.044715);
        var two = TNumber.CreateChecked(2);
        var half = TNumber.One / two;
        var sqrtPiTerm = TNumber.Sqrt(two / TNumber.Pi);

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var xSquared = inputMemory[i] * inputMemory[i];
                var xCubed = inputMemory[i] * outputMemory[i] * inputMemory[i];
                var xPlusMagicXCubed = inputMemory[i] + (magicNumber3 * xCubed);

                // \sqrt{\frac{2}{\pi}}\left(x+0.044715x^{3}\right)
                var tanhArg = sqrtPiTerm * xPlusMagicXCubed;

                // 0.5\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}\left(x+0.044715x^{3}\right)\right)\right)
                var halfTanhTerm = half * (TNumber.One + TNumber.Tanh(tanhArg));

                // \operatorname{sech}\left(\sqrt{\frac{2}{\pi}}\left(x+0.044715x^{3}\right)\right)
                var sechTerm = TNumber.One / TNumber.Cosh(tanhArg);

                // \operatorname{sech}^{2}\left(\sqrt{\frac{2}{\pi}}\left(x+0.044715x^{3}\right)\right)
                var sechSquared = sechTerm * sechTerm;

                // 0.3989422804014327x\left(1+0.134145x^{2}\right)
                var firstMagicTerm = magicNumber1 * inputMemory[i] * (TNumber.One + (magicNumber2 * xSquared));

                outputMemory[i] = (firstMagicTerm * sechSquared) + halfTanhTerm;
            });
    }

    public void SoftPlus<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => outputMemory[i] = TNumber.Log(TNumber.One + TNumber.Exp(inputMemory[i])));
    }

    public void SoftPlusPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, ILogarithmicFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        var logE = TNumber.Log(TNumber.E);

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var exp = TNumber.Exp(inputMemory[i]);

                outputMemory[i] = exp / (TNumber.One + exp) * logE;
            });
    }

    public void SoftSign<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => outputMemory[i] = inputMemory[i] / (TNumber.One + TNumber.Abs(inputMemory[i])));
    }

    public void SoftSignPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;
        var one = TNumber.One;

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var abs = TNumber.Abs(inputMemory[i]);

                outputMemory[i] = one / ((one + abs) * (one + abs));
            });
    }
}