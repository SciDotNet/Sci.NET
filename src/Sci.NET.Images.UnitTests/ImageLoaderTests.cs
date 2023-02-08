// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors.Manipulation;

namespace Sci.NET.Images.UnitTests;

public class ImageLoaderTests
{
    [Fact]
    public void FromFile_ReadsFile()
    {
        var tensor = ImageLoader.FromFile(@"E:\Nudles\old\nude\00e0c6e0-547d-4837-9571-972686dcbff2.jpg");

#pragma warning disable RCS1124
        var img = tensor.Cast<float>() / 255f;
#pragma warning restore RCS1124

        _ = img;

        tensor.Rank.Should().Be(3);
    }
}