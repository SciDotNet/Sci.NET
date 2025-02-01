// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Common;

namespace Sci.NET.Mathematics.Tensors.Manipulation.Implementations;

internal class ReshapeService : IReshapeService
{
    private readonly IGradientAppenderService _gradientAppenderService;

    public ReshapeService()
    {
        _gradientAppenderService = TensorServiceProvider.GetTensorOperationServiceProvider().GetGradientAppenderService();
    }

    public ITensor<TNumber> Reshape<TNumber>(ITensor<TNumber> tensor, Shape shape, bool? overrideRequiresGradient = null)
        where TNumber : unmanaged, INumber<TNumber>
    {
        switch (shape.Count(x => x == -1))
        {
            case > 1:
                throw new ArgumentException("Only one dimension can be inferred from the shape.");
            case 1:
                var totalKnownElements = shape.Where(x => x != -1).Aggregate(1, (x, y) => x * y);
                var inferredDimension = (int)(tensor.Shape.ElementCount / totalKnownElements);
                shape = new Shape(shape.Dimensions.Select(x => x == -1 ? inferredDimension : x).ToArray());
                break;
            default:
                // Nothing to do.
                break;
        }

        tensor.DetachMemory();

        var returnValue = shape.ElementCount != tensor.Shape.ElementCount
            ? throw new ArgumentException("The number of elements in a reshape operation must not change.")
            : new Tensor<TNumber>(tensor, shape, overrideRequiresGradient);

        _gradientAppenderService.AddGradientIfRequired(
            ref returnValue,
            tensor,
            overrideRequiresGradient,
            grad => grad.Reshape(tensor.Shape));

        return returnValue;
    }
}