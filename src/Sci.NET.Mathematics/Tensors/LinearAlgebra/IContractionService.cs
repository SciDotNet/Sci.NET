// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.LinearAlgebra;

/// <summary>
/// An interface containing methods for contracting instances of <see cref="ITensor{TNumber}"/>.
/// </summary>
[PublicAPI]
public interface IContractionService
{
    /// <summary>
    /// Performs a tensor contraction on the specified <paramref name="left"/> and <paramref name="right"/>
    /// <see cref="ITensor{TNumber}"/> instances along the <paramref name="leftIndices"/> and <paramref name="rightIndices"/>.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="leftIndices">The indices of the left operand to contract over.</param>
    /// <param name="rightIndices">The indices of the right operand to contract over.</param>
    /// <param name="overrideRequiresGradient">Overrides the <see cref="ITensor{TNumber}.RequiresGradient"/> property of the result tensor.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the contraction operation.</returns>
    /// <exception cref="ArgumentException">Throws when an argument is invalid.</exception>
    public ITensor<TNumber> Contract<TNumber>(
        ITensor<TNumber> left,
        ITensor<TNumber> right,
        int[] leftIndices,
        int[] rightIndices,
        bool? overrideRequiresGradient = null)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// For vectors, the inner product of two <see cref="ITensor{TNumber}"/>s is calculated,
    /// for higher dimensions then the sum product over the last axes are calculated.
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <param name="overrideRequiresGradient">Overrides the <see cref="ITensor{TNumber}.RequiresGradient"/> property of the result tensor.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the inner product operation.</returns>
    /// <exception cref="ArgumentException">Throws when the operand shapes are incompatible with the
    /// inner product operation.</exception>
    /// <exception cref="InvalidShapeException">The given shapes were not compatible with the inner product operation.</exception>
    public Scalar<TNumber> Inner<TNumber>(Vector<TNumber> left, Vector<TNumber> right, bool? overrideRequiresGradient = null)
        where TNumber : unmanaged, INumber<TNumber>;

    /// <summary>
    /// Calculates the dot product of two <see cref="ITensor{TNumber}"/>s.
    /// <list type="bullet">
    /// <item>If both <paramref name="left"/> and <paramref name="right"/> are 1-D arrays, it is equivalent to <see cref="Inner{TNumber}"/>.</item>
    /// <item>If both <paramref name="left"/> and <paramref name="right"/> are 2-D arrays, it is equivalent to <see cref="MatrixMultiplicationExtensions.MatrixMultiply{TNumber}"/>.</item>
    /// <item>If either <paramref name="left"/> or <paramref name="right"/> is 0-D (scalar), it is equivalent to <see cref="ArithmeticExtensions.Multiply{TNumber}(Sci.NET.Mathematics.Tensors.Scalar{TNumber},Sci.NET.Mathematics.Tensors.Scalar{TNumber})"/></item>
    /// <item>If <paramref name="left"/> is an N-D array and <paramref name="right"/> is a 1-D array, it is a <see cref="Contract{TNumber}"/> operation over the last axis of <paramref name="left"/> and <paramref name="right"/>.</item>
    /// <item>If <paramref name="left"/> is an N-D array and <paramref name="right"/> is an M-D array (where M>=2), it is a <see cref="Contract{TNumber}"/> operation over the last axis of <paramref name="left"/> and the second-to-last axis of <paramref name="right"/>.</item>
    /// </list>
    /// </summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <typeparam name="TNumber">The number type of the <see cref="ITensor{TNumber}"/>.</typeparam>
    /// <returns>The result of the dot product operation.</returns>
    /// <exception cref="InvalidShapeException">Throws when the operand shapes are incompatible with the dot product operation.</exception>
    public ITensor<TNumber> Dot<TNumber>(ITensor<TNumber> left, ITensor<TNumber> right)
        where TNumber : unmanaged, INumber<TNumber>;
}