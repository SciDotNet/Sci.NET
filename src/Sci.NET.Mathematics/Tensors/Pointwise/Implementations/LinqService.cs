// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Pointwise.Implementations;

internal class LinqService : ILinqService
{
    public ITensor<TNumber> Clip<TNumber>(ITensor<TNumber> tensor, TNumber min, TNumber max)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = Tensor.CloneEmpty<ITensor<TNumber>, TNumber>(tensor);

        tensor.Backend.Linq.Clip(tensor, result, min, max);

        return result;
    }
}