// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;

namespace Sci.NET.Mathematics.Tensors.Statistics;

/// <summary>
/// Service for tensor variance.
/// </summary>
[PublicAPI]
public interface IVarianceService
{
    /// <summary>
    /// Calculates the variance of the input tensor.
    /// </summary>
    /// <param name="value">The input tensor.</param>
    /// <param name="axis">The axis to calculate the variance along.</param>
    /// <typeparam name="TNumber">The type of the tensor.</typeparam>
    /// <returns>The variance tensor.</returns>
    public ITensor<TNumber> Variance<TNumber>(ITensor<TNumber> value, int? axis = null)
        where TNumber : unmanaged, IRootFunctions<TNumber>, INumber<TNumber>;
}