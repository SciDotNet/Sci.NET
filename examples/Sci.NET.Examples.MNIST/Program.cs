// See https://aka.ms/new-console-template for more information

using Sci.NET.Datasets.MNIST;
using Sci.NET.Images.Transforms;
using Sci.NET.MachineLearning.NeuralNetworks;
using Sci.NET.MachineLearning.NeuralNetworks.Layers;
using Sci.NET.MachineLearning.NeuralNetworks.Layers.Activations;
using Sci.NET.MachineLearning.NeuralNetworks.Losses;
using Sci.NET.MachineLearning.NeuralNetworks.Optimizers;

using var dataset = new MnistDataset<float>(
    64,
    new NormalizeByFactor<float>(1 / 128.0f));
using var network = new Network<float>();

network.AddLayer(new Flatten<float>());
network.AddLayer(new Dense<float>(784, 512));
network.AddLayer(new ReLU<float>());
network.AddLayer(new Dense<float>(512, 256));
network.AddLayer(new ReLU<float>());
network.AddLayer(new Dense<float>(256, 10));

var optimizer = new Adam<float>(network.Parameters(), 0.1f, 0.9f, 0.999f); var loss = new MeanSquaredError<float>();

var globalStep = 0;

for (var epoch = 0; epoch < 10; epoch++)
{
    foreach (var imageOneHotCategoryBatch in dataset)
    {
        using var output = network.Forward(imageOneHotCategoryBatch.Images);
        using var error = loss.CalculateLoss(imageOneHotCategoryBatch.Labels, output).ToScalar();

        Console.WriteLine($"Epoch: {epoch} Step: {globalStep}, Loss: {error.Value}");

        network.Backward(loss);
        optimizer.Step();

        globalStep++;
        
        imageOneHotCategoryBatch.Dispose();
        output.Dispose();
        error.Dispose();
    }
}