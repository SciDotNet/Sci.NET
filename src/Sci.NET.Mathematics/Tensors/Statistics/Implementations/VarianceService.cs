// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Statistics.Implementations;

internal class VarianceService : IVarianceService
{
    public ITensor<TNumber> Variance<TNumber>(ITensor<TNumber> value, int? axis = null)
        where TNumber : unmanaged, IRootFunctions<TNumber>, INumber<TNumber>
    {
        var concreteAxis = axis ?? 0;

        using var m = new Scalar<TNumber>(TNumber.CreateChecked(value.Shape[concreteAxis]));
        m.To(value.Device);

        var mean = axis is null ? value.Mean() : value.Mean([concreteAxis]).ToVector();

        return axis is null ? mean.Subtract(value).Square().Sum().Divide(m) : mean.Subtract(value).Square().Sum([concreteAxis]).Divide(m);
    }
}