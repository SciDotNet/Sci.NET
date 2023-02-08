// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Sci.NET.DeepLearning.Layers;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.DeepLearning.UnitTests.Layers;

public class DenseTests
{
    [Fact]
    public void DenseLayer_Forward()
    {
        var layer = new Dense<float>(3, 3);

        var input = Tensor.FromArray(
            new Shape(3),
            new float[] { 1, 2, 3 });

        var output = layer.Forward(input);

        output.GetShape().ElementCount.Should().Be(3);
    }
}