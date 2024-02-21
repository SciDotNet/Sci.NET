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
        var logE = TNumber.Log(TNumber.E);

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var softplus = TNumber.Log(TNumber.One + TNumber.Exp(inputMemory[i])) * logE;
                outputMemory[i] = TNumber.Tanh(softplus);
            });
    }

    public void MishPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
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
                // f\left(x\right)=\left(1-\frac{\left(e^{\log\left(1+e^{x}\right)}-e^{-\log\left(1+e^{x}\right)}\right)^{2}}{\left(e^{\log\left(1+e^{x}\right)}+e^{-\log\left(1+e^{x}\right)}\right)^{2}}\right)\cdot\left(\frac{e^{x}}{1+e^{x}}\right)\cdot\log\left(e\right)
                var exp = TNumber.Exp(inputMemory[i]);
                var softplus = TNumber.Log(TNumber.One + exp);
                var expSoftplus = TNumber.Exp(softplus);
                var expNegSoftplus = TNumber.Exp(-softplus);
                var tanhPrimeNumeratorSquared = (expSoftplus - expNegSoftplus) * (expSoftplus - expNegSoftplus);
                var tanhPrimeDenominatorSquared = (expSoftplus + expNegSoftplus) * (expSoftplus + expNegSoftplus);
                var tanhPrimePart = TNumber.One - (tanhPrimeNumeratorSquared / tanhPrimeDenominatorSquared);
                var softPlusPrimePart = exp / (TNumber.One + exp);

                outputMemory[i] = tanhPrimePart * softPlusPrimePart * logE;
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
                if (inputMemory[i] < min || inputMemory[i] > max)
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
        var two = TNumber.CreateChecked(2);

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                outputMemory[i] = inputMemory[i] <= -TNumber.One || inputMemory[i] >= TNumber.One
                    ? TNumber.Zero
                    : (inputMemory[i] + TNumber.One) / two;
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
        var logE = TNumber.Log(TNumber.E);

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var exp = TNumber.One / (TNumber.One + TNumber.Exp(inputMemory[i]));

                outputMemory[i] = exp * logE;
            });
    }

    public void GELU<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;
        var two = TNumber.CreateChecked(2);
        var magicNumber = TNumber.CreateChecked(0.044715);

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var x = inputMemory[i];
                var xSquared = x * x;
                var xCubed = x * xSquared;
                var part1 = TNumber.Sqrt(two / TNumber.Pi);
                var part2 = x + (magicNumber * xCubed);

                outputMemory[i] = x / two * (TNumber.One + TNumber.Tanh(part1 * part2));
            });
    }

    public void GELUPrime<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>, IHyperbolicFunctions<TNumber>, IRootFunctions<TNumber>, IExponentialFunctions<TNumber>
    {
        var inputMemory = (SystemMemoryBlock<TNumber>)value.Memory;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Memory;
        var magicNumber1 = TNumber.CreateChecked(0.044715);
        var magicNumber2 = TNumber.CreateChecked(0.134145);
        var two = TNumber.CreateChecked(2);
        var half = TNumber.CreateChecked(0.5);
        var sqrtPiTerm = TNumber.Sqrt(two / TNumber.Pi);

        _ = LazyParallelExecutor.For(
            0,
            inputMemory.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i =>
            {
                var x = inputMemory[i];

                var tanhArg = sqrtPiTerm * (x + (magicNumber1 * x * x * x));
                var tanhVal = TNumber.Tanh(tanhArg);
                var sechVal = TNumber.One / TNumber.Cosh(tanhArg);

                outputMemory[i] = (half * (TNumber.One + tanhVal)) + (half * x * (TNumber.One - (sechVal * sechVal)) * sqrtPiTerm * (TNumber.One + (magicNumber2 * x * x)));
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