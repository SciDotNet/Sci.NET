// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Common;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.NeuralNetworks.Implementations;

internal class ConvolutionService : IConvolutionService
{
    private readonly IDeviceGuardService _deviceGuardService;

    public ConvolutionService(ITensorOperationServiceProvider tensorOperationServiceProvider)
    {
        _deviceGuardService = tensorOperationServiceProvider.GetDeviceGuardService();
    }

    public Tensor<TNumber> Conv2D<TNumber>(
        Tensor<TNumber> input,
        Tensor<TNumber> kernels,
        int strideX,
        int strideY,
        int paddingX,
        int paddingY,
        int dilationX,
        int dilationY)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var batchSize = input.Shape[0];
        var inputChannels = input.Shape[1];
        var inputHeight = input.Shape[2];
        var inputWidth = input.Shape[3];
        var kernelChannels = kernels.Shape[0];
        var kernelHeight = kernels.Shape[1];
        var kernelWidth = kernels.Shape[2];
        var outputChannels = kernels.Shape[3];
        var outputHeight = ((inputHeight + (2 * paddingY) - (dilationY * (kernelHeight - 1)) - 1) / strideY) + 1;
        var outputWidth = ((inputWidth + (2 * paddingX) - (dilationX * (kernelWidth - 1)) - 1) / strideX) + 1;
        var backend = input.Backend;

        _deviceGuardService.GuardMultiParameterOperation(input.Device, kernels.Device);

        if (inputChannels != kernelChannels)
        {
            throw new ArgumentException("Input and kernel channels must match.", nameof(input));
        }

        var output = new Tensor<TNumber>(new Shape(batchSize, outputChannels, outputHeight, outputWidth), backend);

        backend.NeuralNetworks.Conv2dForward(
            input,
            kernels,
            output,
            strideX,
            strideY,
            paddingX,
            paddingY,
            dilationX,
            dilationY);

        if (input.RequiresGradient)
        {
            ((ITensor<TNumber>)input).AddParent(output, _ => throw new AutoDiffNotSupportedException(nameof(Conv2D)));
        }

        if (kernels.RequiresGradient)
        {
            ((ITensor<TNumber>)kernels).AddParent(output, _ => throw new AutoDiffNotSupportedException(nameof(Conv2D)));
        }

        return output;
    }

    public Tensor<TNumber> Conv2D<TNumber>(
        Tensor<TNumber> input,
        Tensor<TNumber> kernels,
        int stride,
        int padding,
        int dilation)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return Conv2D(
            input,
            kernels,
            stride,
            stride,
            padding,
            padding,
            dilation,
            dilation);
    }
}