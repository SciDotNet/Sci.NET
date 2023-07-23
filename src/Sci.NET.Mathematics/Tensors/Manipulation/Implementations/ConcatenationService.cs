// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.Manipulation.Implementations;

internal class ConcatenationService : IConcatenationService
{
    private readonly IDeviceGuardService _deviceGuardService;

    public ConcatenationService(ITensorOperationServiceProvider tensorOperationServiceProvider)
    {
        _deviceGuardService = tensorOperationServiceProvider.GetDeviceGuardService();
    }

    public Vector<TNumber> Concatenate<TNumber>(ICollection<Scalar<TNumber>> scalars)
        where TNumber : unmanaged, INumber<TNumber>
    {
#pragma warning disable CA2000
        var result = TypeAgnosticConcatenate<Scalar<TNumber>, TNumber>(scalars);
#pragma warning restore CA2000
        return result.AsVector();
    }

    public Matrix<TNumber> Concatenate<TNumber>(ICollection<Vector<TNumber>> vectors)
        where TNumber : unmanaged, INumber<TNumber>
    {
#pragma warning disable CA2000
        var result = TypeAgnosticConcatenate<Vector<TNumber>, TNumber>(vectors);
#pragma warning restore CA2000
        return result.AsMatrix();
    }

    public Tensor<TNumber> Concatenate<TNumber>(ICollection<Matrix<TNumber>> tensors)
        where TNumber : unmanaged, INumber<TNumber>
    {
#pragma warning disable CA2000
        var result = TypeAgnosticConcatenate<Matrix<TNumber>, TNumber>(tensors);
#pragma warning restore CA2000
        return result.AsTensor();
    }

    public Tensor<TNumber> Concatenate<TNumber>(ICollection<Tensor<TNumber>> tensors)
        where TNumber : unmanaged, INumber<TNumber>
    {
#pragma warning disable CA2000
        var result = TypeAgnosticConcatenate<Tensor<TNumber>, TNumber>(tensors);
#pragma warning restore CA2000
        return result.AsTensor();
    }

    private static void EnsureSameShape<TTensor, TNumber>(ICollection<TTensor> tensors)
        where TTensor : ITensor<TNumber>
        where TNumber : unmanaged, INumber<TNumber>
    {
        var shape = tensors.First().Shape;

        if (!tensors.All(t => t.Shape.Dimensions.SequenceEqual(shape.Dimensions)))
        {
            throw new InvalidShapeException("All tensors must have the same shape.");
        }
    }

    private ITensor<TNumber> TypeAgnosticConcatenate<TTensor, TNumber>(ICollection<TTensor> tensors)
        where TTensor : ITensor<TNumber>
        where TNumber : unmanaged, INumber<TNumber>
    {
        EnsureSameShape<TTensor, TNumber>(tensors);
        EnsureOnSameDevice<TTensor, TNumber>(tensors);

        var shape = tensors.First().Shape;
        var newShapeDims = new int[shape.Rank + 1];
        newShapeDims[0] = tensors.Count;

        for (var i = 0; i < shape.Rank; i++)
        {
            newShapeDims[i + 1] = shape[i];
        }

        var newShape = new Shape(newShapeDims);
        var result = new Tensor<TNumber>(newShape, tensors.First().Device.GetTensorBackend());

        for (var i = 0; i < tensors.Count; i++)
        {
            result.Handle.BlockCopy(tensors.ElementAt(i).Handle, 0, i * shape.ElementCount, shape.ElementCount);
        }

        return result;
    }

    private void EnsureOnSameDevice<TTensor, TNumber>(ICollection<TTensor> tensors)
        where TTensor : ITensor<TNumber>
        where TNumber : unmanaged, INumber<TNumber>
    {
        var device = tensors.First().Device;

        foreach (var tensor in tensors)
        {
            _deviceGuardService.GuardBinaryOperation(device, tensor.Device);
        }
    }
}