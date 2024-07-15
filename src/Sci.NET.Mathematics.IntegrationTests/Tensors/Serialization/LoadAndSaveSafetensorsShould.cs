// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Tensors;
using Sci.NET.Tests.Framework.Assertions;

namespace Sci.NET.Mathematics.IntegrationTests.Tensors.Serialization;

public class LoadAndSaveSafetensorsShould
{
    [Fact]
    public void LoadSafetensors()
    {
        // This is not a perfect test, we should be loading from a known file, but this is good enough for now.
        // Arrange
        var tensor1 = Tensor.FromArray<int>(Enumerable.Range(0, 10 * 10 * 10).ToArray()).Reshape(10, 10, 10);
        var tensor2 = Tensor.FromArray<int>(Enumerable.Range(10 * 10 * 10, 9 * 8 * 7).ToArray()).Reshape(9, 8, 7);

        var dictionary = new Dictionary<string, ITensor<int>>
        {
            { "tensor1", tensor1 },
            { "tensor2", tensor2 }
        };

        var id = Guid.NewGuid();

        var tmpDirectory = $"{Environment.CurrentDirectory}\\temp";

        if (!Directory.Exists(tmpDirectory))
        {
            Directory.CreateDirectory(tmpDirectory);
        }

        // Act
        dictionary.SaveSafeTensors($"{tmpDirectory}\\{id}.safetensors");
        var loadedDictionary = Tensor.LoadSafeTensors<int>($"{tmpDirectory}\\{id}.safetensors");

        // Assert
        loadedDictionary.Count.Should().Be(2);
        loadedDictionary["tensor1"].Should().HaveEquivalentElements(tensor1.ToArray());
        loadedDictionary["tensor2"].Should().HaveEquivalentElements(tensor2.ToArray());
    }
}