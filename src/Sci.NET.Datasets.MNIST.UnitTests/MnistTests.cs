// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Tests.Framework.Assertions;

namespace Sci.NET.Datasets.MNIST.UnitTests;

public class MnistTests
{
    [Fact]
    public void LoadMnist_ReturnsMnistDataset()
    {
        var dataset = new MnistDataset<float>(32);

        var batch = dataset.NextBatch();

        batch
            .Images
            .Should()
            .HaveShape(
                32,
                1,
                28,
                28);

        batch
            .Labels
            .Should()
            .HaveShape(32, 10);
    }
}