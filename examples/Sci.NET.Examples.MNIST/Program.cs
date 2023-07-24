// See https://aka.ms/new-console-template for more information

using Sci.NET.Datasets.MNIST;
using Sci.NET.Images.Transforms;
using Sci.NET.MachineLearning.NeuralNetworks;
using Sci.NET.MachineLearning.NeuralNetworks.Layers;
using Sci.NET.MachineLearning.NeuralNetworks.Layers.Activations;
using Sci.NET.MachineLearning.NeuralNetworks.Losses;
using Sci.NET.MachineLearning.NeuralNetworks.Optimizers;

using var dataset = new MnistDataset<float>(32, new NormalizeByFactor<float>(1 / 255f));
using var network = new Network<float>();

network.AddLayer(new Flatten<float>());
network.AddLayer(new Dense<float>(784, 256));
network.AddLayer(new ReLU<float>());
network.AddLayer(new Dense<float>(256, 128));
network.AddLayer(new ReLU<float>());
network.AddLayer(new Dense<float>(128, 10));
network.AddLayer(new Softmax<float>());

var optimizer = new GradientDescent<float>(network.Parameters(), 0.01f);

var globalStep = 0;

foreach (var imageOneHotCategoryBatch in dataset)
{
    var loss = new MeanSquaredError<float>();
    var output = network.Forward(imageOneHotCategoryBatch.Images);
    var error = loss.CalculateLoss(imageOneHotCategoryBatch.Labels, output).AsScalar();

    Console.WriteLine($"Step: {globalStep}, Loss: {error.ScalarValue}");

    network.Backward(error);
    optimizer.Step();

    globalStep++;
}