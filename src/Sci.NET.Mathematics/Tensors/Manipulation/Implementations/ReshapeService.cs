// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Manipulation.Implementations;

internal class ReshapeService : IReshapeService
{
    public ITensor<TNumber> Reshape<TNumber>(ITensor<TNumber> tensor, Shape shape)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return shape.ElementCount != tensor.Shape.ElementCount
            ? throw new ArgumentException("The number of elements in a reshape operation must not change.")
            : new Tensor<TNumber>(tensor, shape);
    }
}