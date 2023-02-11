// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Performance;

namespace Sci.NET.Mathematics.Tensors.Backends.Default;

internal class DefaultMathematicalBackendOperations : IMathematicalBackendOperations
{
    public ITensor<TNumber> Exp<TNumber>(ITensor<TNumber> input)
        where TNumber : unmanaged, INumber<TNumber>, IExponentialFunctions<TNumber>
    {
        var result = new Tensor<TNumber>(new Shape(input.Dimensions));
        var tensorPtr = input.Data;
        var resultPtr = result.Data;

        LazyParallelExecutor.For(
            0,
            input.ElementCount,
            DefaultTensorBackend.ParallelizationThreshold,
            i => resultPtr[i] = TNumber.Exp(tensorPtr[i]));

        return result;
    }
}