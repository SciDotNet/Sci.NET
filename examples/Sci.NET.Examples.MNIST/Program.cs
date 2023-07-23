// See https://aka.ms/new-console-template for more information

using Sci.NET.Datasets.MNIST;
using Sci.NET.Images.Transforms;
using Sci.NET.MachineLearning.NeuralNetworks;
using Sci.NET.MachineLearning.NeuralNetworks.Layers;
using Sci.NET.MachineLearning.NeuralNetworks.Layers.Activations;

using var dataset = new MnistDataset<float>(32, new NormalizeByFactor<float>(1 / 255f));
using var network = new Network<float>();

network.AddLayer(new Flatten<float>());
network.AddLayer(new Dense<float>(784, 784));
network.AddLayer(new ReLU<float>());
network.AddLayer(new Dense<float>(784, 784));
network.AddLayer(new ReLU<float>());
network.AddLayer(new Dense<float>(784, 10));
network.AddLayer(new Softmax<float>());

foreach (var imageOneHotCategoryBatch in dataset)
{
    _ = network.Forward(imageOneHotCategoryBatch.Images);
}