// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Pointwise.Implementations;

internal class LinqService : ILinqService
{
    public TTensor Map<TTensor, TNumber>(TTensor tensor, Func<TNumber, TNumber> func)
        where TTensor : class, ITensor<TNumber>
        where TNumber : unmanaged, INumber<TNumber>
    {
        var result = Tensor.CloneEmpty<TTensor, TNumber>(tensor);

        tensor.Backend.Linq.Map<TTensor, TNumber>(tensor, result, func);

        return result;
    }
}