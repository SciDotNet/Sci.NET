// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;

namespace Sci.NET.MachineLearning.NeuralNetworks.Layers;

/// <summary>
/// Convolutional layer parameters.
/// </summary>
[PublicAPI]
[SuppressMessage(
    "StyleCop.CSharp.OrderingRules",
    "SA1206:Declaration keywords should follow order",
    Justification = "Required keyword should be first.")]
public class Conv2dParameters
{
    /// <summary>
    /// Gets the number of filters.
    /// </summary>
    public required int Filters { get; init; }

    /// <summary>
    /// Gets the kernel size in the x dimension.
    /// </summary>
    public required int KernelSizeX { get; init; }

    /// <summary>
    /// Gets the kernel size in the y dimension.
    /// </summary>
    public required int KernelSizeY { get; init; }

    /// <summary>
    /// Gets the stride in the x dimension.
    /// </summary>
    public required int StrideX { get; init; }

    /// <summary>
    /// Gets the stride in the y dimension.
    /// </summary>
    public required int StrideY { get; init; }

    /// <summary>
    /// Gets the padding in the x dimension.
    /// </summary>
    public required int PaddingX { get; init; }

    /// <summary>
    /// Gets the padding in the y dimension.
    /// </summary>
    public required int PaddingY { get; init; }

    /// <summary>
    /// Gets the dilation in the x dimension.
    /// </summary>
    public required int DilationX { get; init; }

    /// <summary>
    /// Gets the dilation in the y dimension.
    /// </summary>
    public required int DilationY { get; init; }

    /// <summary>
    /// Gets a value indicating whether to use bias.
    /// </summary>
    public required bool UseBias { get; init; }
}