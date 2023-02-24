// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Tensors;
using Sci.NET.Media.Images.PixelFormats;

namespace Sci.NET.Media.Images;

/// <summary>
/// Represents an image.
/// </summary>
/// <typeparam name="TPixel">The type of pixel.</typeparam>
[PublicAPI]
public class Image<TPixel> : IDisposable
    where TPixel : unmanaged, IPixel<TPixel>
{
    private readonly IMemoryBlock<TPixel> _tensor;

    /// <summary>
    /// Initializes a new instance of the <see cref="Image{TPixel}"/> class.
    /// </summary>
    /// <param name="width">The width of the image.</param>
    /// <param name="height">The height of the image.</param>
    public Image(int width, int height)
    {
        _tensor = new SystemMemoryBlock<TPixel>(width * height);
    }

    /// <summary>
    /// Finalizes an instance of the <see cref="Image{TPixel}"/> class.
    /// </summary>
    ~Image()
    {
        Dispose(false);
    }

    /// <summary>
    /// Converts the image to a <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>A <see cref="ITensor{TNumber}"/> representing the current image.</returns>
    /// <exception cref="PlatformNotSupportedException">The current platform is not supported.</exception>
    public Tensor<TNumber> ToTensor<TNumber>()
        where TNumber : unmanaged, INumber<TNumber>
    {
        throw new PlatformNotSupportedException();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes the current instance.
    /// </summary>
    /// <param name="disposing">A value indicating whether the instance is disposing.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            _tensor.Dispose();
        }
    }
}