// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.LinearAlgebra.Implementations;

internal class ReductionService : IReductionService
{
    public ITensor<TNumber> Sum<TNumber>(ITensor<TNumber> tensor, int[]? axes = null, bool keepDims = false)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (axes is null || axes.Length == 0)
        {
            if (!keepDims)
            {
                var result = new Scalar<TNumber>(tensor.Backend);
                tensor.Backend.Reduction.ReduceAddAll(tensor, result);
                return result;
            }
            else
            {
                var resultShape = CalculateResultShape(tensor.Shape.Dimensions, axes, keepDims);
                var result = new Tensor<TNumber>(resultShape, tensor.Backend);
                tensor.Backend.Reduction.ReduceAddAllKeepDims(tensor, result);
                return result;
            }
        }

        if (!keepDims)
        {
            var resultShape = CalculateResultShape(tensor.Shape.Dimensions, axes, keepDims);
            var result = new Tensor<TNumber>(resultShape, tensor.Backend);
            tensor.Backend.Reduction.ReduceAddAxis(tensor, axes, result);
            return result;
        }
        else
        {
            var resultShape = CalculateResultShape(tensor.Shape.Dimensions, axes, keepDims);
            var result = new Tensor<TNumber>(resultShape, tensor.Backend);
            tensor.Backend.Reduction.ReduceAddAxisKeepDims(tensor, axes, result);
            return result;
        }
    }

    private static Shape CalculateResultShape(int[] shape, int[]? axes, bool keepDims)
    {
        var axisSet = axes is not null ? new HashSet<int>(axes) : new HashSet<int>();

        var resultShapeDimensions = new int[shape.Length];

        for (var i = 0; i < shape.Length; i++)
        {
#pragma warning disable RCS1238
            resultShapeDimensions[i] = axisSet.Contains(i) ? keepDims ? 1 : 0 : shape[i];
#pragma warning restore RCS1238
        }

        return new Shape(resultShapeDimensions.Where(dim => dim != 0).ToArray());
    }
}