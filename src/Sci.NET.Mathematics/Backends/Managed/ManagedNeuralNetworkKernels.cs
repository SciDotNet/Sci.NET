// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Concurrency;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends.Managed;

internal class ManagedNeuralNetworkKernels : INeuralNetworkKernels
{
    public void Conv2dForward<TNumber>(
        Tensor<TNumber> input,
        Tensor<TNumber> kernels,
        Tensor<TNumber> result,
        int strideX,
        int strideY,
        int paddingX,
        int paddingY,
        int dilationX,
        int dilationY)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var batchSize = input.Shape[0];
        var inChannels = input.Shape[1];
        var inHeight = input.Shape[2];
        var inWidth = input.Shape[3];
        var kernelHeight = kernels.Shape[1];
        var kernelWidth = result.Shape[2];
        var outChannels = result.Shape[1];
        var outHeight = result.Shape[2];
        var outWidth = result.Shape[3];
        var inputMemory = (SystemMemoryBlock<TNumber>)input.Handle;
        var kernelMemory = (SystemMemoryBlock<TNumber>)kernels.Handle;
        var outputMemory = (SystemMemoryBlock<TNumber>)result.Handle;

        LazyParallelExecutor.For(
            0,
            batchSize,
            0,
            outHeight,
            0,
            (n, oh) =>
            {
                for (var ow = 0; ow < outWidth; ow++)
                {
                    for (var oc = 0; oc < outChannels; oc++)
                    {
                        var sum = TNumber.Zero;

                        for (var kh = 0; kh < kernelHeight; kh++)
                        {
                            for (var kw = 0; kw < kernelWidth; kw++)
                            {
                                for (var ic = 0; ic < inChannels; ic++)
                                {
                                    var ih = (oh * strideY) + (kh * dilationY);
                                    var iw = (ow * strideX) + (kw * dilationX);

                                    if (ih < 0 || ih >= inHeight || iw < 0 || iw >= inWidth)
                                    {
                                        continue;
                                    }

                                    var inputIndex = (n * input.Shape.Strides[0]) +
                                                     (ic * input.Shape.Strides[1]) +
                                                     (ih * input.Shape.Strides[2]) +
                                                     (iw * input.Shape.Strides[3]);

                                    var kernelIndex = (oc * kernels.Shape.Strides[0]) +
                                                      (ic * kernels.Shape.Strides[1]) +
                                                      (kh * kernels.Shape.Strides[2]) +
                                                      (kw * kernels.Shape.Strides[3]);

                                    sum += inputMemory[inputIndex] *
                                           kernelMemory[kernelIndex];
                                }
                            }
                        }

                        var outputIndex = (n * result.Shape.Strides[0]) +
                                          (oc * result.Shape.Strides[1]) +
                                          (oh * result.Shape.Strides[2]) +
                                          (ow * result.Shape.Strides[3]);
                        outputMemory[outputIndex] = sum;
                    }
                }
            });
    }

    public void Conv2dBackward<TNumber>(
        Tensor<TNumber> input,
        Tensor<TNumber> kernels,
        Tensor<TNumber> dOutput,
        Tensor<TNumber> dInput,
        Tensor<TNumber> dKernel,
        int strideX,
        int strideY,
        int paddingX,
        int paddingY,
        int dilationX,
        int dilationY)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var batchSize = input.Shape.Dimensions[0];
        var inChannels = input.Shape.Dimensions[1];
        var inHeight = input.Shape.Dimensions[2];
        var inWidth = input.Shape.Dimensions[3];
        var outChannels = kernels.Shape.Dimensions[0];
        var kernelHeight = kernels.Shape.Dimensions[2];
        var kernelWidth = kernels.Shape.Dimensions[3];
        var outHeight = dOutput.Shape.Dimensions[2];
        var outWidth = dOutput.Shape.Dimensions[3];
        var inputMemory = (SystemMemoryBlock<TNumber>)input.Handle;
        var kernelMemory = (SystemMemoryBlock<TNumber>)kernels.Handle;
        var dOutputMemory = (SystemMemoryBlock<TNumber>)dOutput.Handle;
        var dInputMemory = (SystemMemoryBlock<TNumber>)dInput.Handle;
        var dKernelMemory = (SystemMemoryBlock<TNumber>)dKernel.Handle;

        LazyParallelExecutor.For(
            0,
            batchSize,
            0,
            outHeight,
            0,
            (n, oh) =>
            {
                for (var ow = 0; ow < outWidth; ow++)
                {
                    for (var oc = 0; oc < outChannels; oc++)
                    {
                        var dOutputIndex = (n * dOutput.Shape.Strides[0]) +
                                           (oc * dOutput.Shape.Strides[1]) +
                                           (oh * dOutput.Shape.Strides[2]) +
                                           (ow * dOutput.Shape.Strides[3]);

                        var dOutputValue = dOutputMemory[dOutputIndex];

                        for (var kh = 0; kh < kernelHeight; kh++)
                        {
                            for (var kw = 0; kw < kernelWidth; kw++)
                            {
                                for (var ic = 0; ic < inChannels; ic++)
                                {
                                    var ih = (oh * strideY) + (kh * dilationY);
                                    var iw = (ow * strideX) + (kw * dilationX);

                                    if (ih < 0 || ih >= inHeight || iw < 0 || iw >= inWidth)
                                    {
                                        continue;
                                    }

                                    var inputIndex = (n * input.Shape.Strides[0]) +
                                                     (ic * input.Shape.Strides[1]) +
                                                     (ih * input.Shape.Strides[2]) +
                                                     (iw * input.Shape.Strides[3]);

                                    var kernelIndex = (oc * kernels.Shape.Strides[0]) +
                                                      (ic * kernels.Shape.Strides[1]) +
                                                      (kh * kernels.Shape.Strides[2]) +
                                                      (kw * kernels.Shape.Strides[3]);

                                    dInputMemory[inputIndex] +=
                                        kernelMemory[kernelIndex] * dOutputValue;

                                    dKernelMemory[kernelIndex] +=
                                        inputMemory[inputIndex] * dOutputValue;
                                }
                            }
                        }
                    }
                }
            });
    }
}