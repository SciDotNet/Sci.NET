// See https://aka.ms/new-console-template for more information

using Sci.NET.Datasets.MNIST;
using Sci.NET.Images.Transforms;
using Sci.NET.MachineLearning.NeuralNetworks;
using Sci.NET.MachineLearning.NeuralNetworks.Layers;
using Sci.NET.MachineLearning.NeuralNetworks.Layers.Activations;
using Sci.NET.MachineLearning.NeuralNetworks.Losses;
using Sci.NET.MachineLearning.NeuralNetworks.Optimizers;



using var dataset = new MnistDataset<float>(
    32,
    new NormalizeByFactor<float>(1 / 255f),
    new ClipToRange<float>(0.00001f, 0.99999f));
using var network = new Network<float>();

network.AddLayer(new Flatten<float>());
network.AddLayer(new Dense<float>(784, 384));
network.AddLayer(new ReLU<float>());
network.AddLayer(new Dense<float>(384, 128));
network.AddLayer(new ReLU<float>());
network.AddLayer(new Dense<float>(128, 64));
network.AddLayer(new ReLU<float>());
network.AddLayer(new Dense<float>(64, 32));
network.AddLayer(new ReLU<float>());
network.AddLayer(new Dense<float>(32, 16));
network.AddLayer(new ReLU<float>());
network.AddLayer(new Dense<float>(16, 10));
network.AddLayer(new Sigmoid<float>());

var optimizer = new Adam<float>(network.Parameters(), 0.02f, 0.9f, 0.999f); var loss = new MeanSquaredError<float>();

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