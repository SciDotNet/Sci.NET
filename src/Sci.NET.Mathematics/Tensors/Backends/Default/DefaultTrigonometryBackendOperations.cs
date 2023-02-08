// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Performance;

namespace Sci.NET.Mathematics.Tensors.Backends.Default;

internal class DefaultTrigonometryBackendOperations : ITrigonometryBackendOperations
{
    public ITensor<TNumber> Sin<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(new Shape(tensor.Dimensions));
        var tensorPtr = tensor.Data;
        var resultPtr = result.Data;

        LazyParallelExecutor.For(
            0,
            tensor.ElementCount,
            DefaultTensorBackend.ParallelizationThreshold,
            i => resultPtr[i] = TNumber.Sin(tensorPtr[i]));

        return result;
    }

    public ITensor<TNumber> Cos<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(new Shape(tensor.Dimensions));
        var tensorPtr = tensor.Data;
        var resultPtr = result.Data;

        LazyParallelExecutor.For(
            0,
            tensor.ElementCount,
            DefaultTensorBackend.ParallelizationThreshold,
            i => resultPtr[i] = TNumber.Cos(tensorPtr[i]));
        return result;
    }

    public ITensor<TNumber> Tan<TNumber>(ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>, ITrigonometricFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(new Shape(tensor.Dimensions));
        var tensorPtr = tensor.Data;
        var resultPtr = result.Data;

        LazyParallelExecutor.For(
            0,
            tensor.ElementCount,
            DefaultTensorBackend.ParallelizationThreshold,
            i => resultPtr[i] = TNumber.Tan(tensorPtr[i]));

        return result;
    }
}