using Sci.NET.Mathematics.Tensors;

// Add to matrices with the same shape.
using var tensor1 = Tensor.FromArray<int>(new int[,] { { 1, 2 }, { 3, 4 } });
using var tensor2 = Tensor.FromArray<int>(new int[,] { { 5, 6 }, { 7, 8 } });
using var result1 = tensor1.Add(tensor2);

// Subtract a vector from a matrix using broadcasting.
using var tensor3 = Tensor.FromArray<int>(new int[] { 6, 6 });
using var result2 = tensor1.Subtract(tensor3);
using var result4 = tensor3.Subtract(tensor1);

// Output the results.
Console.WriteLine(result1); // Output: [[6, 8], [10, 12]]
Console.WriteLine(result2); // Output: [[-5, -4], [-3, -2]]
Console.WriteLine(result4); // Output: [[5, 4], [3, 2]]