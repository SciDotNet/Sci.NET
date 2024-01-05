// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Serialization;

public class SaveShould
{
    [Fact]
    public void Save_GivenTensor()
    {
        var tensor = Tensor
            .Random
            .Uniform(new Shape(100, 100), 0.0f, 1.0f);

        var id = Guid.NewGuid();

        var tmpDirectory = $"{Environment.CurrentDirectory}\\temp";

        if (!Directory.Exists(tmpDirectory))
        {
            Directory.CreateDirectory(tmpDirectory);
        }

        tensor.Save($"{tmpDirectory}\\{id}-uncompressed.bin");
        tensor.SaveCompressed($"{tmpDirectory}\\{id}-compressed.bin");

        var loadedUncompressed = Tensor.Load<float>($"{tmpDirectory}\\{id}-uncompressed.bin");
        var loadedCompressed = Tensor.LoadCompressed<float>($"{tmpDirectory}\\{id}-compressed.bin");

        var loadedUncompressedArray = loadedUncompressed.Memory.ToArray();
        var loadedCompressedArray = loadedCompressed.Memory.ToArray();

        loadedUncompressedArray.Should().BeEquivalentTo(loadedCompressedArray);
    }
}