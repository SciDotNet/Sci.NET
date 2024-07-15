// See https://aka.ms/new-console-template for more information

using Sci.NET.Mathematics.Tensors;

using var matrixA = Tensor.FromArray<int>(new int[,] { { 1, 2 }, { 3, 4 } }).ToMatrix();
using var matrixB = Tensor.FromArray<int>(new int[,] { { 5, 6 }, { 7, 8 } }).ToMatrix();

var result = matrixA.MatrixMultiply(matrixB);

Console.WriteLine(result); // Output: [[19, 22], [43, 50]]

using var randomMatrixA = Tensor.Random.Uniform(new Shape(100,500), 0.0f, 1.0f).ToMatrix();
using var randomMatrixB = Tensor.Random.Uniform(new Shape(500,200), 0.0f, 1.0f).ToMatrix();

var randomResult = randomMatrixA.MatrixMultiply(randomMatrixB);

Console.WriteLine(randomResult); // Output: Matrix with shape [100, 200]