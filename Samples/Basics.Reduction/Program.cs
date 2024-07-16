
using Sci.NET.Mathematics.Tensors;

var tensor = Tensor.FromArray<int>(new int[,] { { 1, 2 }, { 3, 4 } });

var result1 = tensor.Sum([0]);
var result2 = tensor.Sum([1]);

Console.WriteLine(result1); // Output: [4, 6]
Console.WriteLine(result2); // Output: [3, 7]