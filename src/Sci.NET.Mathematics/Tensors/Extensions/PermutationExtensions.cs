// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;

// ReSharper disable once CheckNamespace
#pragma warning disable IDE0130 // API accessibility
namespace Sci.NET.Mathematics.Tensors;
#pragma warning restore IDE0130

/// <summary>
/// Extension methods to permute a <see cref="ITensor{TNumber}"/>.
/// </summary>
[PublicAPI]
public static class PermutationExtensions
{
    /// <summary>
    /// Permutes the <see cref="ITensor{TNumber}"/>, rearranging the indices
    /// in the order passed as a parameter.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to permute.</param>
    /// <param name="permutation">The new order of the indices.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The permuted <see cref="ITensor{TNumber}"/>.</returns>
    /// <exception cref="ArgumentException">Throws when the <paramref name="permutation"/>
    /// indices are invalid.</exception>
    [DebuggerStepThrough]
    public static ITensor<TNumber> Permute<TNumber>(this ITensor<TNumber> tensor, int[] permutation)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPermutationService()
            .Permute(tensor, permutation);
    }

    /// <summary>
    /// Transposes the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to transpose.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The transposed <see cref="ITensor{TNumber}"/>.</returns>
    public static ITensor<TNumber> Transpose<TNumber>(this ITensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var transposeIndices = Enumerable.Range(0, tensor.Shape.Rank).Reverse().ToArray();

        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPermutationService()
            .Permute(tensor, transposeIndices);
    }

    /// <summary>
    /// Transposes the <see cref="ITensor{TNumber}"/> using the specified permutation.
    /// </summary>
    /// <param name="tensor">The <see cref="ITensor{TNumber}"/> to transpose.</param>
    /// <param name="permutation">The permutation to use for the transpose.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The transposed <see cref="ITensor{TNumber}"/>.</returns>
    /// <remarks>This is an alias for <see cref="Permute{TNumber}(ITensor{TNumber},int[])"/>.</remarks>
    public static ITensor<TNumber> Transpose<TNumber>(this ITensor<TNumber> tensor, int[] permutation)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPermutationService()
            .Permute(tensor, permutation);
    }

    /// <summary>
    /// Transposes the <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <param name="matrix">The <see cref="Matrix{TNumber}"/> to transpose.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
    /// <returns>The transposed <see cref="Matrix{TNumber}"/>.</returns>
    public static Matrix<TNumber> Transpose<TNumber>(this Matrix<TNumber> matrix)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPermutationService()
            .Permute(matrix, new[] { 1, 0 })
            .ToMatrix();
    }

    /// <summary>
    /// Transposes the <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <param name="tensor">The <see cref="Tensor{TNumber}"/> to transpose.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="Tensor{TNumber}"/>.</typeparam>
    /// <returns>The transposed <see cref="Tensor{TNumber}"/>.</returns>
    public static Tensor<TNumber> Transpose<TNumber>(this Tensor<TNumber> tensor)
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TensorServiceProvider
            .GetTensorOperationServiceProvider()
            .GetPermutationService()
            .Permute(tensor, Enumerable.Range(0, tensor.Shape.Rank).Reverse().ToArray())
            .ToTensor();
    }
}