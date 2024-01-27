// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.Reduction.Implementations;

internal class ReductionService : IReductionService
{
    public ITensor<TNumber> Sum<TNumber>(ITensor<TNumber> tensor, int[]? axes = null, bool keepDims = false)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (axes is null || axes.Length == 0 || tensor.Shape.Rank - axes.Length <= 0)
        {
            var result = new Scalar<TNumber>(tensor.Backend);
            tensor.Backend.Reduction.ReduceAddAll(tensor, result);

            if (!keepDims)
            {
                return result;
            }

            var resultTensor = result.Broadcast(tensor.Shape);
            result.Dispose();
            return resultTensor;
        }
        else
        {
            if (axes.Length > tensor.Shape.Dimensions.Length)
            {
                throw new InvalidShapeException($"The number of axes to sum over cannot exceed the number of dimensions in shape {tensor.Shape}.");
            }

            if (axes.Any(x => x < 0 || x >= tensor.Shape.Rank))
            {
                throw new InvalidShapeException($"The axes to sum over must be within the bounds of the tensor with shape {tensor.Shape}.");
            }

            var resultShape = CalculateResultShape(tensor.Shape.Dimensions, axes, keepDims);
            var result = new Tensor<TNumber>(resultShape, tensor.Backend);
            tensor.Backend.Reduction.ReduceAddAxis(tensor, axes, result);

            if (!keepDims)
            {
                return result;
            }

            var resultTensor = result.Broadcast(tensor.Shape);
            result.Dispose();

            return resultTensor;
        }
    }

    private static Shape CalculateResultShape(int[] shape, int[]? axes, bool keepDims)
    {
        var axisSet = axes is not null ? new HashSet<int>(axes) : new HashSet<int>();

        var resultShapeDimensions = new int[shape.Length];

        for (var i = 0; i < shape.Length; i++)
        {
#pragma warning disable IDE0045
            if (axisSet.Contains(i))
#pragma warning restore IDE0045
            {
                resultShapeDimensions[i] = keepDims ? 1 : 0;
            }
            else
            {
                resultShapeDimensions[i] = shape[i];
            }
        }

        return new Shape(resultShapeDimensions.Where(dim => dim != 0).ToArray());
    }
}