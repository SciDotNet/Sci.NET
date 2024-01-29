// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedCastingKernels : ICastingKernels
{
    public void Cast<TIn, TOut>(Scalar<TIn> input, Scalar<TOut> output)
        where TIn : unmanaged, INumber<TIn>
        where TOut : unmanaged, INumber<TOut>
    {
        var inputMemoryBlock = (SystemMemoryBlock<TIn>)input.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TOut>)output.Memory;

        resultMemoryBlock[0] = TOut.CreateChecked(inputMemoryBlock[0]);
    }

    public void Cast<TIn, TOut>(Tensors.Vector<TIn> input, Tensors.Vector<TOut> output)
        where TIn : unmanaged, INumber<TIn>
        where TOut : unmanaged, INumber<TOut>
    {
        var inputMemoryBlock = (SystemMemoryBlock<TIn>)input.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TOut>)output.Memory;

        _ = LazyParallelExecutor.For(
            0,
            resultMemoryBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            i => resultMemoryBlock[i] = TOut.CreateChecked(inputMemoryBlock[i]));
    }

    public void Cast<TIn, TOut>(Matrix<TIn> input, Matrix<TOut> output)
        where TIn : unmanaged, INumber<TIn>
        where TOut : unmanaged, INumber<TOut>
    {
        var inputMemoryBlock = (SystemMemoryBlock<TIn>)input.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TOut>)output.Memory;

        _ = LazyParallelExecutor.For(
            0,
            output.Rows,
            0,
            output.Columns,
            ManagedTensorBackend.ParallelizationThreshold,
            (i, j) => resultMemoryBlock[(i * output.Columns) + j] =
                TOut.CreateChecked(inputMemoryBlock[(i * input.Columns) + j]));
    }

    public void Cast<TIn, TOut>(Tensor<TIn> input, Tensor<TOut> output)
        where TIn : unmanaged, INumber<TIn>
        where TOut : unmanaged, INumber<TOut>
    {
        var inputMemoryBlock = (SystemMemoryBlock<TIn>)input.Memory;
        var resultMemoryBlock = (SystemMemoryBlock<TOut>)output.Memory;

        _ = LazyParallelExecutor.For(
            0,
            inputMemoryBlock.Length,
            ManagedTensorBackend.ParallelizationThreshold / 2,
            i => resultMemoryBlock[i] = TOut.CreateChecked(inputMemoryBlock[i]));
    }
}