// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Performance;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.Backends.Default;

internal class DefaultArithmeticBackendOperations : IArithmeticBackendOperations
{
    public ITensor<TNumber> Add<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftData = left.Data;
        var rightData = right.Data;

        if (left.IsScalar)
        {
            return Add(leftData[0], right);
        }

        if (right.IsScalar)
        {
            return Add(rightData[0], left);
        }

        if (left.ElementCount != right.ElementCount)
        {
            throw new InvalidShapeException(
                $"The shapes {left.GetShape()} and {right.GetShape()} are not " +
                "compatible for addition. They must have the same number of elements.");
        }

        var result = new Tensor<TNumber>(left.GetShape());
        var resultPtr = result.Data;

        LazyParallelExecutor.For(
            0,
            left.ElementCount,
            DefaultTensorBackend.ParallelizationThreshold,
            i => resultPtr[i] = leftData[i] + rightData[i]);

        return result;
    }

    public ITensor<TNumber> Add<TNumber>(TNumber left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(right.GetShape());
        var rightData = right.Data;
        var resultPtr = result.Data;

        LazyParallelExecutor.For(
            0,
            right.ElementCount,
            DefaultTensorBackend.ParallelizationThreshold,
            i => resultPtr[i] = left + rightData[i]);

        return result;
    }

    public ITensor<TNumber> Subtract<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var leftData = left.Data;
        var rightData = right.Data;

        if (left.IsScalar)
        {
            return Subtract(leftData[0], right);
        }

        if (right.IsScalar)
        {
            return Subtract(left, rightData[0]);
        }

        if (left.ElementCount != right.ElementCount)
        {
            throw new InvalidShapeException(
                $"The shapes {left.GetShape()} and {right.GetShape()} are not " +
                "compatible for addition. They must have the same number of elements.");
        }

        var result = new Tensor<TNumber>(left.GetShape());
        var resultPtr = result.Data;

        LazyParallelExecutor.For(
            0,
            left.ElementCount,
            DefaultTensorBackend.ParallelizationThreshold,
            i => resultPtr[i] = leftData[i] - rightData[i]);

        return result;
    }

    public ITensor<TNumber> Subtract<TNumber>(TNumber left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(right.GetShape());
        var rightData = right.Data;
        var resultPtr = result.Data;

        LazyParallelExecutor.For(
            0,
            right.ElementCount,
            DefaultTensorBackend.ParallelizationThreshold,
            i => resultPtr[i] = left - rightData[i]);

        return result;
    }

    public ITensor<TNumber> Subtract<TNumber>(ITensor<TNumber> left, TNumber right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(left.GetShape());
        var leftData = left.Data;
        var resultPtr = result.Data;

        LazyParallelExecutor.For(
            0,
            left.ElementCount,
            DefaultTensorBackend.ParallelizationThreshold,
            i => resultPtr[i] = leftData[i] - right);

        return result;
    }

    public ITensor<TNumber> Negate<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(tensor.GetShape());
        var leftData = tensor.Data;
        var resultPtr = result.Data;

        LazyParallelExecutor.For(
            0,
            tensor.ElementCount,
            DefaultTensorBackend.ParallelizationThreshold,
            i => resultPtr[i] = TNumber.Zero - leftData[i]);

        return result;
    }

    public ITensor<TNumber> Divide<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(left.GetShape());
        var leftData = left.Data;
        var rightData = right.Data;
        var resultPtr = result.Data;

        if (left.ElementCount != right.ElementCount)
        {
            throw new InvalidShapeException(
                $"The shapes {left.GetShape()} and {right.GetShape()} are not " +
                "compatible for addition. They must have the same number of elements.");
        }

        LazyParallelExecutor.For(
            0,
            left.ElementCount,
            DefaultTensorBackend.ParallelizationThreshold,
            i => resultPtr[i] = leftData[i] / rightData[i]);

        return result;
    }

    public ITensor<TNumber> Divide<TNumber>(TNumber left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(right.GetShape());
        var rightData = right.Data;
        var resultPtr = result.Data;

        LazyParallelExecutor.For(
            0,
            right.ElementCount,
            DefaultTensorBackend.ParallelizationThreshold,
            i => resultPtr[i] = left / rightData[i]);

        return result;
    }

    public ITensor<TNumber> Divide<TNumber>(ITensor<TNumber> left, TNumber right)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = new Tensor<TNumber>(left.GetShape());
        var leftData = left.Data;
        var resultPtr = result.Data;

        LazyParallelExecutor.For(
            0,
            left.ElementCount,
            DefaultTensorBackend.ParallelizationThreshold,
            i => resultPtr[i] = leftData[i] / right);

        return result;
    }
}