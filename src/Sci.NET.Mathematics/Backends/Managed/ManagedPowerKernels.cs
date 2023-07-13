// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedPowerKernels : IPowerKernels
{
    public void Pow<TNumber>(Scalar<TNumber> value, Scalar<TNumber> power, Scalar<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        var valueBlock = (SystemMemoryBlock<TNumber>)value.Handle;
        var powerBlock = (SystemMemoryBlock<TNumber>)power.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        resultBlock[0] = TNumber.Pow(valueBlock[0], powerBlock[0]);
    }

    public void Pow<TNumber>(Tensors.Vector<TNumber> value, Scalar<TNumber> power, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        var valueBlock = (SystemMemoryBlock<TNumber>)value.Handle;
        var powerBlock = (SystemMemoryBlock<TNumber>)power.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            valueBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Pow(valueBlock[0], powerBlock[i]));
    }

    public void Pow<TNumber>(Matrix<TNumber> value, Scalar<TNumber> power, Matrix<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        var valueBlock = (SystemMemoryBlock<TNumber>)value.Handle;
        var powerBlock = (SystemMemoryBlock<TNumber>)power.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            valueBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Pow(valueBlock[0], powerBlock[i]));
    }

    public void Pow<TNumber>(Tensor<TNumber> value, Scalar<TNumber> power, Tensor<TNumber> result)
        where TNumber : unmanaged, IPowerFunctions<TNumber>, INumber<TNumber>
    {
        var valueBlock = (SystemMemoryBlock<TNumber>)value.Handle;
        var powerBlock = (SystemMemoryBlock<TNumber>)power.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            valueBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Pow(valueBlock[0], powerBlock[i]));
    }

    public void Square<TNumber>(Scalar<TNumber> value, Scalar<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var valueBlock = (SystemMemoryBlock<TNumber>)value.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        resultBlock[0] = TNumber.Abs(valueBlock[0] * valueBlock[0]);
    }

    public void Square<TNumber>(Tensors.Vector<TNumber> value, Tensors.Vector<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var valueBlock = (SystemMemoryBlock<TNumber>)value.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            valueBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Abs(valueBlock[i] * valueBlock[i]));
    }

    public void Square<TNumber>(Matrix<TNumber> value, Matrix<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var valueBlock = (SystemMemoryBlock<TNumber>)value.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            valueBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Abs(valueBlock[i] * valueBlock[i]));
    }

    public void Square<TNumber>(Tensor<TNumber> value, Tensor<TNumber> result)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var valueBlock = (SystemMemoryBlock<TNumber>)value.Handle;
        var resultBlock = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            valueBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold,
            i => resultBlock[i] = TNumber.Abs(valueBlock[i] * valueBlock[i]));
    }
}