// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.Tensors.NeuralNetworks;

/// <summary>
/// Describes a convolutional layer.
/// </summary>
[PublicAPI]
public class ConvolutionalDescriptor
{
    /// <summary>
    /// Gets the shape of the kernel.
    /// </summary>
    public int[] KernelShape { get; }

    /// <summary>
    /// Gets the strides of the convolution.
    /// </summary>
    public int[] Stride { get; }

    /// <summary>
    /// Gets the padding of the convolution.
    /// </summary>
    public int[] Padding { get; }
}