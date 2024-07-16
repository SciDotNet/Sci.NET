using Sci.NET.Mathematics.Tensors;

// Multiply two matrices with shapes [2, 2] and [2, 2].
using var matrixA = Tensor.FromArray<int>(new int[,] { { 1, 2 }, { 3, 4 } }).ToMatrix();
using var matrixB = Tensor.FromArray<int>(new int[,] { { 5, 6 }, { 7, 8 } }).ToMatrix();

// Multiply two matrices.
var result = matrixA.MatrixMultiply(matrixB);

// Output: [[19, 22], [43, 50]]
Console.WriteLine(result);

// Multiply two random matrices with shapes [100, 500] and [500, 200].
using var randomMatrixA = Tensor.Random.Uniform(new Shape(100, 500), 0.0f, 1.0f).ToMatrix();
using var randomMatrixB = Tensor.Random.Uniform(new Shape(500, 200), 0.0f, 1.0f).ToMatrix();

// Multiply two the two matrices.
var randomResult = randomMatrixA.MatrixMultiply(randomMatrixB);

// Output: Matrix with shape [100, 200]
Console.WriteLine(randomResult);