// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.MachineLearning.NeuralNetworks;
using Sci.NET.MachineLearning.NeuralNetworks.Layers;
using Sci.NET.MachineLearning.NeuralNetworks.Layers.Activations;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.MachineLearning.UnitTests;

public class NetworkTests
{
    [Fact]
    public void BasicFunctionalityTest()
    {
        var layer1 = new Dense<float>(256, 128);
        var relu1 = new ReLU<float>();
        var layer2 = new Dense<float>(128, 64);
        var relu2 = new ReLU<float>();
        var layer3 = new Dense<float>(64, 32);
        var relu3 = new ReLU<float>();
        var layer4 = new Dense<float>(32, 16);
        var relu4 = new ReLU<float>();
        var layer5 = new Dense<float>(16, 8);
        var relu5 = new ReLU<float>();
        var layer6 = new Dense<float>(8, 4);
        var relu6 = new ReLU<float>();
        var layer7 = new Dense<float>(4, 2);
        var relu7 = new ReLU<float>();
        var layer8 = new Dense<float>(2, 1);
        var relu8 = new ReLU<float>();
        var network = new Network<float>();

        network.AddLayer(layer1);
        network.AddLayer(relu1);
        network.AddLayer(layer2);
        network.AddLayer(relu2);
        network.AddLayer(layer3);
        network.AddLayer(relu3);
        network.AddLayer(layer4);
        network.AddLayer(relu4);
        network.AddLayer(layer5);
        network.AddLayer(relu5);
        network.AddLayer(layer6);
        network.AddLayer(relu6);
        network.AddLayer(layer7);
        network.AddLayer(relu7);
        network.AddLayer(layer8);
        network.AddLayer(relu8);

        var result = network.Forward(Tensor.Zeros<float>(1, 256));

        var handle = result.Handle;
        handle.IsDisposed.Should().BeFalse();

        result.Should().NotBeNull();
    }
}