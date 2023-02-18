// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Common.Performance;
using Sci.NET.Mathematics.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Sci.NET.Images;

/// <summary>
/// Provides methods to load images.
/// </summary>
[PublicAPI]
public static class ImageLoader
{
    /// <summary>
    /// Loads an image from the specified path.
    /// </summary>
    /// <param name="path">The path to the image.</param>
    /// <returns>A <see cref="ITensor{TNumber}"/> with the image data.</returns>
    /// <exception cref="NotSupportedException">The image format is not supported.</exception>
    public static ITensor<byte> FromFile(string path)
    {
        using var image = Image.Load(path, out var format);

        return image switch
        {
            Image<Rgb24> rgb24 => LoadRgb24(rgb24),
            Image<Rgba32> rgba32 => LoadRgba32(rgba32),
            Image<Rgb48> rgb48 => LoadRgb24(rgb48.CloneAs<Rgb24>()),
            Image<Rgba64> rgba64 => LoadRgba32(rgba64.CloneAs<Rgba32>()),
            Image<Bgr24> bgr24 => LoadRgb24(bgr24.CloneAs<Rgb24>()),
            Image<Bgra32> bgra32 => LoadRgba32(bgra32.CloneAs<Rgba32>()),
            Image<L8> l8 => LoadBw8(l8),
            Image<L16> l16 => LoadBw8(l16.CloneAs<L8>()),
            _ => throw new NotSupportedException("Unsupported image format: " + format.Name)
        };
    }

    /// <summary>
    /// Loads an image from the specified path.
    /// </summary>
    /// <param name="path">The path to the image.</param>
    /// <returns>A <see cref="ITensor{TNumber}"/> with the image data.</returns>
    /// <exception cref="NotSupportedException">The image format is not supported.</exception>
    public static ITensor<byte> FromFileRemovingAlpha(string path)
    {
        using var image = Image.Load(path, out var format);

        return image switch
        {
            Image<Rgb24> rgb24 => LoadRgb24(rgb24),
            Image<Rgba32> rgba32 => LoadRgb24(rgba32.CloneAs<Rgb24>()),
            Image<Rgb48> rgb48 => LoadRgb24(rgb48.CloneAs<Rgb24>()),
            Image<Rgba64> rgba64 => LoadRgb24(rgba64.CloneAs<Rgb24>()),
            Image<Bgr24> bgr24 => LoadRgb24(bgr24.CloneAs<Rgb24>()),
            Image<Bgra32> bgra32 => LoadRgb24(bgra32.CloneAs<Rgb24>()),
            Image<L8> l8 => LoadBw8(l8),
            Image<L16> l16 => LoadBw8(l16.CloneAs<L8>()),
            _ => throw new NotSupportedException("Unsupported image format: " + format.Name)
        };
    }

    private static ITensor<byte> LoadRgb24(Image<Rgb24> image)
    {
        var tensor = new Tensor<byte>(new Shape(image.Width, image.Height, 3));

        // ReSharper disable AccessToDisposedClosure
        LazyParallelExecutor.For(
            0,
            image.Width,
            10000,
            i =>
            {
                for (var j = 0; j < image.Height; j++)
                {
                    var pixel = image[(int)i, j];
                    var index = ((i * image.Width) + j) * 3;

                    tensor.Data[index + 0] = pixel.R;
                    tensor.Data[index + 1] = pixel.G;
                    tensor.Data[index + 2] = pixel.B;
                }
            });

        image.Dispose();

        return tensor;
    }

    private static ITensor<byte> LoadRgba32(Image<Rgba32> image)
    {
        var tensor = new Tensor<byte>(new Shape(image.Width, image.Height, 4));

        // ReSharper disable AccessToDisposedClosure
        LazyParallelExecutor.For(
            0,
            image.Width,
            10000,
            i =>
            {
                for (var j = 0; j < image.Height; j++)
                {
                    var pixel = image[(int)i, j];
                    var index = ((i * image.Width) + j) * 4;

                    tensor.Data[index + 0] = pixel.R;
                    tensor.Data[index + 1] = pixel.G;
                    tensor.Data[index + 2] = pixel.B;
                    tensor.Data[index + 3] = pixel.A;
                }
            });

        image.Dispose();

        return tensor;
    }

    private static ITensor<byte> LoadBw8(Image<L8> image)
    {
        var tensor = new Tensor<byte>(new Shape(image.Width, image.Height, 1));

        // ReSharper disable AccessToDisposedClosure
        LazyParallelExecutor.For(
            0,
            image.Width,
            10000,
            i =>
            {
                for (var j = 0; j < image.Height; j++)
                {
                    var pixel = image[(int)i, j];
                    var index = (i * image.Width) + j;

                    tensor.Data[index] = pixel.PackedValue;
                }
            });

        image.Dispose();

        return tensor;
    }
}