// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Backends.Managed;
using Sci.NET.Mathematics.Tensors;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;

namespace Sci.NET.Images;

/// <summary>
/// Provides image processing functionality.
/// </summary>
public static class Image
{
    /// <summary>
    /// Loads an image from the specified path in RGB24 format.
    /// </summary>
    /// <param name="path">The path to the image.</param>
    /// <returns>A <see cref="ITensor{TNumber}"/> containing the image.</returns>
    [PublicAPI]
    public static Tensor<byte> LoadRgb24(string path)
    {
        var image = SixLabors.ImageSharp.Image.Load<Rgb24>(new DecoderOptions(), path);
        var memory = image.Frames.RootFrame.PixelBuffer.MemoryGroup.TotalLength;

        using var pixelMemoryBlock = new SystemMemoryBlock<Rgb24>(memory);
        image.CopyPixelDataTo(pixelMemoryBlock.AsSpan());

        var convertedMemory = pixelMemoryBlock.DangerousReinterpretCast<byte>();

        return new Tensor<byte>(
            convertedMemory,
            new Shape(3, image.Height, image.Width),
            ManagedTensorBackend.Instance);
    }

    /// <summary>
    /// Loads an image from the specified path in RGB48 format.
    /// </summary>
    /// <param name="path">The path to the image.</param>
    /// <returns>A <see cref="ITensor{TNumber}"/> containing the image.</returns>
    [PublicAPI]
    public static Tensor<ushort> LoadRgb48(string path)
    {
        var image = SixLabors.ImageSharp.Image.Load<Rgb48>(new DecoderOptions(), path);
        var memory = image.Frames.RootFrame.PixelBuffer.MemoryGroup.TotalLength;

        using var pixelMemoryBlock = new SystemMemoryBlock<Rgb48>(memory);
        image.CopyPixelDataTo(pixelMemoryBlock.AsSpan());

        var convertedMemory = pixelMemoryBlock.DangerousReinterpretCast<ushort>();

        return new Tensor<ushort>(
            convertedMemory,
            new Shape(3, image.Height, image.Width),
            ManagedTensorBackend.Instance);
    }

    /// <summary>
    /// Loads an image from the specified path in RGBA32 format.
    /// </summary>
    /// <param name="path">The path to the image.</param>
    /// <returns>A <see cref="ITensor{TNumber}"/> containing the image.</returns>
    [PublicAPI]
    public static Tensor<byte> LoadRgba32(string path)
    {
        var image = SixLabors.ImageSharp.Image.Load<Rgba32>(new DecoderOptions(), path);
        var memory = image.Frames.RootFrame.PixelBuffer.MemoryGroup.TotalLength;

        using var pixelMemoryBlock = new SystemMemoryBlock<Rgba32>(memory);
        image.CopyPixelDataTo(pixelMemoryBlock.AsSpan());

        var convertedMemory = pixelMemoryBlock.DangerousReinterpretCast<byte>();

        return new Tensor<byte>(
            convertedMemory,
            new Shape(4, image.Height, image.Width),
            ManagedTensorBackend.Instance);
    }

    /// <summary>
    /// Loads an image from the specified path in RGBA64 format.
    /// </summary>
    /// <param name="path">The path to the image.</param>
    /// <returns>A <see cref="ITensor{TNumber}"/> containing the image.</returns>
    [PublicAPI]
    public static Tensor<ushort> LoadRgba64(string path)
    {
        var image = SixLabors.ImageSharp.Image.Load<Rgba64>(new DecoderOptions(), path);
        var memory = image.Frames.RootFrame.PixelBuffer.MemoryGroup.TotalLength;

        using var pixelMemoryBlock = new SystemMemoryBlock<Rgba64>(memory);
        image.CopyPixelDataTo(pixelMemoryBlock.AsSpan());

        var convertedMemory = pixelMemoryBlock.DangerousReinterpretCast<ushort>();

        return new Tensor<ushort>(
            convertedMemory,
            new Shape(4, image.Height, image.Width),
            ManagedTensorBackend.Instance);
    }

    /// <summary>
    /// Loads an image from the specified path in 8bpp grayscale format.
    /// </summary>
    /// <param name="path">The path to the image.</param>
    /// <returns>A <see cref="ITensor{TNumber}"/> containing the image.</returns>
    [PublicAPI]
    public static Tensor<byte> LoadL8(string path)
    {
        var image = SixLabors.ImageSharp.Image.Load<L8>(new DecoderOptions(), path);
        var memory = image.Frames.RootFrame.PixelBuffer.MemoryGroup.TotalLength;

        using var pixelMemoryBlock = new SystemMemoryBlock<L8>(memory);
        image.CopyPixelDataTo(pixelMemoryBlock.AsSpan());

        var convertedMemory = pixelMemoryBlock.DangerousReinterpretCast<byte>();

        return new Tensor<byte>(
            convertedMemory,
            new Shape(1, image.Height, image.Width),
            ManagedTensorBackend.Instance);
    }

    /// <summary>
    /// Loads an image from the specified path in 16bpp grayscale format.
    /// </summary>
    /// <param name="path">The path to the image.</param>
    /// <returns>A <see cref="ITensor{TNumber}"/> containing the image.</returns>
    [PublicAPI]
    public static Tensor<ushort> LoadL16(string path)
    {
        var image = SixLabors.ImageSharp.Image.Load<L16>(new DecoderOptions(), path);
        var memory = image.Frames.RootFrame.PixelBuffer.MemoryGroup.TotalLength;

        using var pixelMemoryBlock = new SystemMemoryBlock<L16>(memory);
        image.CopyPixelDataTo(pixelMemoryBlock.AsSpan());

        var convertedMemory = pixelMemoryBlock.DangerousReinterpretCast<ushort>();

        return new Tensor<ushort>(
            convertedMemory,
            new Shape(1, image.Height, image.Width),
            ManagedTensorBackend.Instance);
    }
}