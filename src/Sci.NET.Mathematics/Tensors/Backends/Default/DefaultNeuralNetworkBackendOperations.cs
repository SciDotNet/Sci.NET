// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Performance;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.Backends.Default;

internal class DefaultNeuralNetworkBackendOperations : INeuralNetworkBackendOperations
{
    public ITensor<TNumber> Conv2d<TNumber>(
        ITensor<TNumber> input,
        ITensor<TNumber> kernel,
        int strideX,
        int strideY,
        int dilationX,
        int dilationY)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (input.Rank != 3)
        {
            throw new InvalidShapeException($"The input must be of rank 3 but was rank {input.Rank}", input.GetShape());
        }

        if (kernel.Rank != 4)
        {
            throw new InvalidShapeException(
                $"The kernel must be of rank 4 but was rank {kernel.Rank}",
                kernel.GetShape());
        }

        if (kernel.Dimensions[2] != input.Dimensions[2])
        {
            throw new InvalidShapeException(
                $"The number of input channels must be {kernel.Dimensions[2]} but {input.Dimensions[2]}",
                input.GetShape());
        }

        var inputHeight = input.Dimensions[0];
        var inputWidth = input.Dimensions[1];
        var inputChannels = input.Dimensions[2];
        var kernelHeight = kernel.Dimensions[0];
        var kernelWidth = kernel.Dimensions[1];
        var outputChannels = kernel.Dimensions[3];
        var outputHeight = ((inputHeight - (dilationY * (kernelHeight - 1)) - 1) / strideY) + 1;
        var outputWidth = ((inputWidth - (dilationX * (kernelWidth - 1)) - 1) / strideX) + 1;
        var output = new Tensor<TNumber>(new Shape(outputWidth, outputHeight, outputChannels));
        var inputData = input.Data;
        var kernelData = kernel.Data;
        var outputData = output.Data;

        LazyParallelExecutor.For(
            0,
            outputHeight,
            0,
            outputWidth,
            100,
            (y, x) =>
            {
                for (var k = 0; k < outputChannels; k++)
                {
                    var sum = TNumber.Zero;

                    for (var i = 0; i < kernelHeight; i++)
                    {
                        var inputY = (y * strideX) + (i * dilationX);

                        if (inputY >= inputHeight)
                        {
                            break;
                        }

                        for (var j = 0; j < kernelWidth; j++)
                        {
                            var inputX = (x * strideY) + (j * dilationY);

                            if (inputX >= inputWidth)
                            {
                                break;
                            }

                            for (var c = 0; c < inputChannels; c++)
                            {
                                sum += inputData[(inputY * inputWidth * inputChannels) + (inputX * inputChannels) + c] *
                                       kernelData[(i * kernelWidth * inputChannels * outputChannels) +
                                                  (j * inputChannels * outputChannels) + (c * outputChannels) + k];
                            }
                        }
                    }

                    outputData[(y * outputWidth * outputChannels) + (x * outputChannels) + k] = sum;
                }
            });

        return output;
    }
}