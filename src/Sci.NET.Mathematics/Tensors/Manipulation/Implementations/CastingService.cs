// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.Manipulation.Implementations;

internal class CastingService : ICastingService
{
    public Scalar<TOut> Cast<TIn, TOut>(Scalar<TIn> input)
        where TIn : unmanaged, INumber<TIn>
        where TOut : unmanaged, INumber<TOut>
    {
        var result = new Scalar<TOut>(input.Backend);

        input.Backend.Casting.Cast(input, result);

        if (input.RequiresGradient)
        {
            throw new AutoDiffNotSupportedException(nameof(Cast));
        }

        return result;
    }

    public Vector<TOut> Cast<TIn, TOut>(Vector<TIn> input)
        where TIn : unmanaged, INumber<TIn>
        where TOut : unmanaged, INumber<TOut>
    {
        var result = new Vector<TOut>(input.Length, input.Backend);

        input.Backend.Casting.Cast(input, result);

        if (input.RequiresGradient)
        {
            throw new AutoDiffNotSupportedException(nameof(Cast));
        }

        return result;
    }

    public Matrix<TOut> Cast<TIn, TOut>(Matrix<TIn> input)
        where TIn : unmanaged, INumber<TIn>
        where TOut : unmanaged, INumber<TOut>
    {
        var result = new Matrix<TOut>(input.Rows, input.Columns, input.Backend);

        input.Backend.Casting.Cast(input, result);

        if (input.RequiresGradient)
        {
            throw new AutoDiffNotSupportedException(nameof(Cast));
        }

        return result;
    }

    public Tensor<TOut> Cast<TIn, TOut>(Tensor<TIn> input)
        where TIn : unmanaged, INumber<TIn>
        where TOut : unmanaged, INumber<TOut>
    {
        if (typeof(TIn) == typeof(TOut))
        {
            var inputMemoryBlock = (SystemMemoryBlock<TIn>)input.Memory;
            var newMemoryBlock = inputMemoryBlock
                .Copy()
                .ToSystemMemory()
                .DangerousReinterpretCast<TOut>();

            if (input.RequiresGradient)
            {
                throw new AutoDiffNotSupportedException(nameof(Cast));
            }

            return new Tensor<TOut>(newMemoryBlock, input.Shape, input.Backend, input.RequiresGradient);
        }

        var result = new Tensor<TOut>(input.Shape, input.Backend);

        input.Backend.Casting.Cast(input, result);

        if (input.RequiresGradient)
        {
            throw new AutoDiffNotSupportedException(nameof(Cast));
        }

        return result;
    }
}