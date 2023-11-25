// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.CUDA.Tensors;
using Sci.NET.Mathematics.Tensors;

var left = Tensor.FromArray<int>(new int[] { 1, 2, 3, 4 }).ToVector();
var right = Tensor.FromArray<int>(new int[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }).ToMatrix();

var result = left.Add(right);

Parallel.For(
    0,
    1000,
    i =>
    {
        using var tensor = Tensor.Random.Uniform<float, CudaComputeDevice>(
            new Shape(50, 50, 10),
            0,
            1,
            4);
    });