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
    /// <param name="transparency">I value indicating whether transparency should be loaded.</param>
    /// <returns>A <see cref="ITensor{TNumber}"/> with the image data.</returns>
    public static ITensor<byte> FromFile(string path, bool transparency = false)
    {
        return transparency ? LoadWithTransparency(path) : LoadWithoutTransparency(path);
    }

    private static ITensor<byte> LoadWithTransparency(string path)
    {
        var image = Image.Load(path);
        var tensor = new Tensor<byte>(new Shape(image.Width, image.Height, 3));
        using var bitmap = image.CloneAs<Rgba32>();

        // ReSharper disable AccessToDisposedClosure
        LazyParallelExecutor.For(
            0,
            image.Width,
            10000,
            i =>
            {
                for (var j = 0; j < image.Height; j++)
                {
                    var pixel = bitmap[(int)i, j];
                    var index = ((i * image.Height) + j) * 4;

                    tensor.Data[index] = pixel.R;
                    tensor.Data[index + 1] = pixel.G;
                    tensor.Data[index + 2] = pixel.B;
                    tensor.Data[index + 3] = pixel.A;
                }
            });

        image.Dispose();

        return tensor;
    }

    private static ITensor<byte> LoadWithoutTransparency(string path)
    {
        var image = Image.Load(path);
        var tensor = new Tensor<byte>(new Shape(image.Width, image.Height, 3));
        using var bitmap = image.CloneAs<Rgb24>();

        // ReSharper disable AccessToDisposedClosure
        LazyParallelExecutor.For(
            0,
            image.Width,
            10000,
            i =>
            {
                for (var j = 0; j < image.Height; j++)
                {
                    var pixel = bitmap[(int)i, j];
                    var index = ((i * image.Height) + j) * 3;

                    tensor.Data[index] = pixel.R;
                    tensor.Data[index + 1] = pixel.G;
                    tensor.Data[index + 2] = pixel.B;
                }
            });

        image.Dispose();

        return tensor;
    }
}