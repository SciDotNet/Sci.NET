// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.Manipulation.Implementations;

internal class ConcatenationService : IConcatenationService
{
    private readonly IDeviceGuardService _deviceGuardService;
    private readonly IGradientAppenderService _gradientAppenderService;

    public ConcatenationService()
    {
        _deviceGuardService = TensorServiceProvider.GetTensorOperationServiceProvider().GetDeviceGuardService();
        _gradientAppenderService = TensorServiceProvider.GetTensorOperationServiceProvider().GetGradientAppenderService();
    }

    public Vector<TNumber> Concatenate<TNumber>(ICollection<Scalar<TNumber>> scalars)
        where TNumber : unmanaged, INumber<TNumber>
    {
#pragma warning disable CA2000
        var result = TypeAgnosticConcatenate<Scalar<TNumber>, TNumber>(scalars);
#pragma warning restore CA2000

        return result.ToVector();
    }

    public Matrix<TNumber> Concatenate<TNumber>(ICollection<Vector<TNumber>> vectors)
        where TNumber : unmanaged, INumber<TNumber>
    {
#pragma warning disable CA2000
        var result = TypeAgnosticConcatenate<Vector<TNumber>, TNumber>(vectors);
#pragma warning restore CA2000

        return result.ToMatrix();
    }

    public Tensor<TNumber> Concatenate<TNumber>(ICollection<Matrix<TNumber>> matrices)
        where TNumber : unmanaged, INumber<TNumber>
    {
#pragma warning disable CA2000
        var result = TypeAgnosticConcatenate<Matrix<TNumber>, TNumber>(matrices);
#pragma warning restore CA2000

        return result.ToTensor();
    }

    public Tensor<TNumber> Concatenate<TNumber>(ICollection<Tensor<TNumber>> tensors)
        where TNumber : unmanaged, INumber<TNumber>
    {
#pragma warning disable CA2000
        var result = TypeAgnosticConcatenate<Tensor<TNumber>, TNumber>(tensors);
#pragma warning restore CA2000

        return result.ToTensor();
    }

    private static void EnsureSameShape<TTensor, TNumber>(ICollection<TTensor> tensors)
        where TTensor : ITensor<TNumber>
        where TNumber : unmanaged, INumber<TNumber>
    {
        var shape = tensors.First().Shape;

        if (!tensors.All(t => t.Shape.Dimensions.SequenceEqual(shape.Dimensions)))
        {
            throw new InvalidShapeException($"All tensors must have the same shape, but were {string.Join(',', tensors.Select(x => x.Shape.ToString()))}.");
        }
    }

#pragma warning disable CA1859
    private ITensor<TNumber> TypeAgnosticConcatenate<TTensor, TNumber>(ICollection<TTensor> tensors)
#pragma warning restore CA1859
        where TTensor : ITensor<TNumber>
        where TNumber : unmanaged, INumber<TNumber>
    {
        EnsureSameShape<TTensor, TNumber>(tensors);
        var backend = _deviceGuardService.GuardMultiParameterOperation(tensors.Select(x => x.Device).ToArray());

        var shape = tensors.First().Shape;
        var newShapeDims = new int[shape.Rank + 1];
        newShapeDims[0] = tensors.Count;

        for (var i = 0; i < shape.Rank; i++)
        {
            newShapeDims[i + 1] = shape[i];
        }

        var newShape = new Shape(newShapeDims);
        var result = new Tensor<TNumber>(newShape, backend);

        for (var i = 0; i < tensors.Count; i++)
        {
            result.Memory.BlockCopyFrom(tensors.ElementAt(i).Memory, 0, i * shape.ElementCount, shape.ElementCount);
        }

        foreach (var tensor in tensors)
        {
            _gradientAppenderService.AddGradientIfRequired(
                ref result,
                tensor,
                null,
                _ => throw new AutoDiffNotSupportedException(nameof(Concatenate)));
        }

        return result;
    }
}