// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedPowerKernels : IPowerKernels
{
    public void Pow<TNumber>(ITensor<TNumber> value, Scalar<TNumber> power, ITensor<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        var valueBlock = (SystemMemoryBlock<TNumber>)value.Memory;
        var powerBlock = (SystemMemoryBlock<TNumber>)power.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            valueBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Pow(valueBlock[i], powerBlock[0]));
    }

    public void PowDerivative<TNumber>(ITensor<TNumber> value, Scalar<TNumber> power, ITensor<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        var valueBlock = (SystemMemoryBlock<TNumber>)value.Memory;
        var powerBlock = (SystemMemoryBlock<TNumber>)power.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            valueBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = powerBlock[0] * TNumber.Pow(valueBlock[i], powerBlock[0] - TNumber.One));
    }

    public void Square<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var valueBlock = (SystemMemoryBlock<TNumber>)value.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            valueBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Abs(valueBlock[i] * valueBlock[i]));
    }

    public void Exp<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, IExponentialFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        var valueBlock = (SystemMemoryBlock<TNumber>)value.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            valueBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Exp(valueBlock[i]));
    }

    public void Log<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        var valueBlock = (SystemMemoryBlock<TNumber>)value.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            valueBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = resultBlock[i] >= TNumber.Zero ? TNumber.Log(valueBlock[i]) : TNumber.NaN);
    }

    public void LogDerivative<TNumber>(ITensor<TNumber> value, ITensor<TNumber> result)
        where TNumber : unmanaged, ILogarithmicFunctions<TNumber>, IFloatingPointIeee754<TNumber>, INumber<TNumber>
    {
        var valueBlock = (SystemMemoryBlock<TNumber>)value.Memory;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Memory;

        _ = LazyParallelExecutor.For(
            0,
            valueBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = valueBlock[i] <= TNumber.Zero ? TNumber.NaN : TNumber.One / valueBlock[i]);
    }
}