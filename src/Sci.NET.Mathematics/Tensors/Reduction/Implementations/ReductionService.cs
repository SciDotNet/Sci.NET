// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.Reduction.Implementations;

internal class ReductionService : IReductionService
{
    public ITensor<TNumber> ReduceToShape<TNumber>(ITensor<TNumber> tensor, Shape targetShape)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var currentShape = tensor.Shape;
        var axesToReduce = new List<int>();

        for (var i = 0; i < currentShape.Rank; i++)
        {
            if (currentShape[i] != targetShape[i])
            {
                axesToReduce.Add(i);
            }
        }

        if (axesToReduce.Count > 0)
        {
            return tensor.Sum(axesToReduce.ToArray());
        }

        throw new InvalidShapeException($"The tensor with shape {currentShape} cannot be reduced to the target shape {targetShape}.");
    }

    public bool CanReduceToShape<TNumber>(ITensor<TNumber> tensor, Shape shape)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var originalShape = tensor.Shape.Dimensions;
        var targetShape = shape.Dimensions;

        var originalRank = originalShape.Length;
        var targetRank = targetShape.Length;

        var j = targetRank - 1;
        for (var i = originalRank - 1; i >= 0; i--)
        {
            if (j < 0)
            {
                continue;
            }

            if (originalShape[i] == targetShape[j])
            {
                j--;
            }
            else if (originalShape[i] <= targetShape[j])
            {
                return false;
            }
        }

        return j < 0;
    }

    public ITensor<TNumber> Sum<TNumber>(ITensor<TNumber> tensor, int[]? axes = null, bool keepDims = false)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (axes is null || axes.Length == 0 || tensor.Shape.Rank - axes.Length <= 0)
        {
            var result = new Scalar<TNumber>(tensor.Backend, requiresGradient: tensor.RequiresGradient);
            tensor.Backend.Reduction.ReduceAddAll(tensor, result);

            if (!keepDims)
            {
                ((ITensor<TNumber>)result).AddParent(
                    tensor,
                    grad => grad.Broadcast(tensor.Shape));

                return result;
            }

            var resultTensor = result.Broadcast(tensor.Shape);
            result.Dispose();

            if (tensor.RequiresGradient)
            {
                resultTensor.AddParent(
                    tensor,
                    grad =>
                    {
                        var outShape = grad.Shape.ExpandDims(axes ?? Array.Empty<int>());
                        return grad.Broadcast(outShape);
                    });
            }

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
            var result = new Tensor<TNumber>(resultShape, tensor.Backend, requiresGradient: tensor.RequiresGradient);
            tensor.Backend.Reduction.ReduceAddAxis(tensor, axes, result);

            if (keepDims)
            {
                if (tensor.RequiresGradient)
                {
                    ((ITensor<TNumber>)result).AddParent(
                        tensor,
                        grad =>
                        {
                            var outShape = grad.Shape.ExpandDims(axes);
                            return grad.Broadcast(outShape);
                        });
                }

                return result;
            }

            if (tensor.RequiresGradient)
            {
                ((ITensor<TNumber>)result).AddParent(
                    tensor,
                    grad => grad.Broadcast(tensor.Shape));
            }

            return result;
        }
    }

    public ITensor<TNumber> Mean<TNumber>(ITensor<TNumber> tensor, int[]? axes = null, bool keepDims = false)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (axes is null || axes.Length == 0 || tensor.Shape.Rank - axes.Length <= 0)
        {
            var result = new Scalar<TNumber>(tensor.Backend, requiresGradient: tensor.RequiresGradient);
            tensor.Backend.Reduction.ReduceMeanAll(tensor, result);

            if (!keepDims)
            {
                if (tensor.RequiresGradient)
                {
                    ((ITensor<TNumber>)result).AddParent(
                        tensor,
                        grad =>
                        {
                            var gradExpanded = grad.Broadcast(tensor.Shape);
                            var scale = TNumber.One / TNumber.CreateChecked(tensor.Shape.ElementCount);
                            using var scaleTensor = new Scalar<TNumber>(scale, tensor.Backend, requiresGradient: false);
                            return gradExpanded.Multiply(scaleTensor);
                        });
                }

                return result;
            }

            var resultTensor = result.Broadcast(tensor.Shape);
            result.Dispose();

            if (tensor.RequiresGradient)
            {
                resultTensor.AddParent(
                    tensor,
                    grad =>
                    {
                        var gradExpanded = grad.Broadcast(tensor.Shape);
                        var scale = TNumber.One / TNumber.CreateChecked(tensor.Shape.ElementCount);
                        using var scaleTensor = new Scalar<TNumber>(scale, tensor.Backend, requiresGradient: false);
                        return gradExpanded.Multiply(scaleTensor);
                    });
            }

            return resultTensor;
        }
        else
        {
            if (axes.Length > tensor.Shape.Dimensions.Length)
            {
                throw new InvalidShapeException($"The number of axes to find the mean over cannot exceed the number of dimensions in shape {tensor.Shape}.");
            }

            if (axes.Any(x => x < 0 || x >= tensor.Shape.Rank))
            {
                throw new InvalidShapeException($"The axes to find the mean over must be within the bounds of the tensor with shape {tensor.Shape}.");
            }

            var resultShape = CalculateResultShape(tensor.Shape.Dimensions, axes, keepDims);
            var result = new Tensor<TNumber>(resultShape, tensor.Backend, requiresGradient: tensor.RequiresGradient);
            tensor.Backend.Reduction.ReduceMeanAxis(tensor, axes, result);

            if (!keepDims)
            {
                if (tensor.RequiresGradient)
                {
                    ((ITensor<TNumber>)result).AddParent(
                        tensor,
                        grad =>
                        {
                            var gradExpanded = grad.Broadcast(tensor.Shape);
                            var reductionSize = axes.Aggregate(1, (prod, axis) => prod * tensor.Shape.Dimensions[axis]);
                            var scale = TNumber.One / TNumber.CreateChecked(reductionSize);
                            using var scaleTensor = new Scalar<TNumber>(scale, tensor.Backend, requiresGradient: false);
                            return gradExpanded.Multiply(scaleTensor);
                        });
                }

                return result;
            }

            var resultTensor = result.Broadcast(tensor.Shape);
            result.Dispose();

            if (tensor.RequiresGradient)
            {
                resultTensor.AddParent(
                    tensor,
                    grad =>
                    {
                        // Expand dimensions for the gradient, broadcast, and scale
                        var gradExpanded = grad.Broadcast(tensor.Shape);
                        var reductionSize = axes.Aggregate(1, (prod, axis) => prod * tensor.Shape.Dimensions[axis]);
                        var scale = TNumber.One / TNumber.CreateChecked(reductionSize);
                        using var scaleTensor = new Scalar<TNumber>(scale, tensor.Backend, requiresGradient: false);
                        return gradExpanded.Multiply(scaleTensor);
                    });
            }

            return resultTensor;
        }
    }

    public ITensor<TNumber> Max<TNumber>(ITensor<TNumber> tensor, int[]? axes = null, bool keepDims = false)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (axes is null || axes.Length == 0 || tensor.Shape.Rank - axes.Length <= 0)
        {
            var result = new Scalar<TNumber>(tensor.Backend);
            tensor.Backend.Reduction.ReduceMaxAll(tensor, result);

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
                throw new InvalidShapeException($"The number of axes to find the max over cannot exceed the number of dimensions in shape {tensor.Shape}.");
            }

            if (axes.Any(x => x < 0 || x >= tensor.Shape.Rank))
            {
                throw new InvalidShapeException($"The axes to find the max over must be within the bounds of the tensor with shape {tensor.Shape}.");
            }

            var resultShape = CalculateResultShape(tensor.Shape.Dimensions, axes, keepDims);
            var result = new Tensor<TNumber>(resultShape, tensor.Backend);
            tensor.Backend.Reduction.ReduceMaxAxis(tensor, axes, result);

            if (!keepDims)
            {
                return result;
            }

            var resultTensor = result.Broadcast(tensor.Shape);
            result.Dispose();

            if (tensor.RequiresGradient)
            {
                resultTensor.AddParent(
                    tensor,
                    grad =>
                    {
                        var gradExpanded = grad.Broadcast(tensor.Shape);
                        var reductionSize = axes.Aggregate(1, (prod, axis) => prod * tensor.Shape.Dimensions[axis]);
                        var scale = TNumber.One / TNumber.CreateChecked(reductionSize);
                        using var scaleTensor = new Scalar<TNumber>(scale, tensor.Backend, requiresGradient: false);
                        return gradExpanded.Multiply(scaleTensor);
                    });
            }

            return resultTensor;
        }
    }

    public ITensor<TNumber> Min<TNumber>(ITensor<TNumber> tensor, int[]? axes = null, bool keepDims = false)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (axes is null || axes.Length == 0 || tensor.Shape.Rank - axes.Length <= 0)
        {
            var result = new Scalar<TNumber>(tensor.Backend);
            tensor.Backend.Reduction.ReduceMinAll(tensor, result);

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
                throw new InvalidShapeException($"The number of axes to find the min over cannot exceed the number of dimensions in shape {tensor.Shape}.");
            }

            if (axes.Any(x => x < 0 || x >= tensor.Shape.Rank))
            {
                throw new InvalidShapeException($"The axes to find the min over must be within the bounds of the tensor with shape {tensor.Shape}.");
            }

            var resultShape = CalculateResultShape(tensor.Shape.Dimensions, axes, keepDims);
            var result = new Tensor<TNumber>(resultShape, tensor.Backend);
            tensor.Backend.Reduction.ReduceMinAxis(tensor, axes, result);

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