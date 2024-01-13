// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using FluentAssertions.Execution;
using FluentAssertions.Primitives;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Tests.Framework.Assertions;

/// <summary>
/// Assertions for <see cref="ITensor{TNumber}" />.
/// </summary>
/// <typeparam name="TNumber">The type of the tensor.</typeparam>
public class TensorAssertions<TNumber> : ReferenceTypeAssertions<ITensor<TNumber>, TensorAssertions<TNumber>>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TensorAssertions{TNumber}"/> class.
    /// </summary>
    /// <param name="subject">The tensor to create assertions for.</param>
    public TensorAssertions(ITensor<TNumber> subject)
        : base(subject)
    {
    }

    /// <inheritdoc />
    protected override string Identifier => "tensor";

    /// <summary>
    /// Asserts that the tensor has the given shape.
    /// </summary>
    /// <param name="shape">The expected shape.</param>
    /// <returns>A <see cref="AndConstraint{TAssertions}" /> object.</returns>
    public AndConstraint<TensorAssertions<TNumber>> HaveShape(params int[] shape)
    {
        _ = Execute
            .Assertion
            .BecauseOf(string.Empty, Array.Empty<object>())
            .Given(() => Subject.Shape)
            .ForCondition(tensorShape => tensorShape.SequenceEqual(shape))
            .FailWith("Expected tensor to have shape {0}{reason}, but found {1}.", shape, Subject.Shape);

        return new AndConstraint<TensorAssertions<TNumber>>(this);
    }

    /// <summary>
    /// Asserts that the tensor has the given shape.
    /// </summary>
    /// <param name="shape">The expected shape.</param>
    /// <returns>A <see cref="AndConstraint{TAssertions}" /> object.</returns>
    public AndConstraint<TensorAssertions<TNumber>> HaveShape(Shape shape)
    {
        _ = Execute
            .Assertion
            .BecauseOf(string.Empty, Array.Empty<object>())
            .Given(() => Subject.Shape)
            .ForCondition(tensorShape => tensorShape.SequenceEqual(shape))
            .FailWith("Expected tensor to have shape {0}{reason}, but found {1}.", shape, Subject.Shape);

        return new AndConstraint<TensorAssertions<TNumber>>(this);
    }

    /// <summary>
    /// Asserts that the tensor has the given shape.
    /// </summary>
    /// <param name="shape">The expected shape.</param>
    /// <returns>A <see cref="AndConstraint{TAssertions}" /> object.</returns>
    public AndConstraint<TensorAssertions<TNumber>> HaveEquivalentElements(Array shape)
    {
        _ = Execute
            .Assertion
            .BecauseOf(string.Empty, Array.Empty<object>())
            .Given(() => Subject.ToArray())
            .ForCondition(tensorElements => AreEquivalentElements(tensorElements, shape))
            .FailWith("Expected tensor to have elements {0}{reason}, but found {1}.", shape, Subject.ToArray());

        return new AndConstraint<TensorAssertions<TNumber>>(this);
    }

    private static bool AreEquivalentElements(Array tensor1, Array tensor2)
    {
        if (tensor1.Rank != tensor2.Rank)
        {
            return false;
        }

        for (var dimension = 0; dimension < tensor1.Rank; dimension++)
        {
            if (tensor1.GetLength(dimension) != tensor2.GetLength(dimension))
            {
                return false;
            }
        }

        return CompareElements(tensor1, tensor2);
    }

    private static bool CompareElements(Array tensor1, Array tensor2)
    {
        var indices = new int[tensor1.Rank];
        var sizes = new int[tensor1.Rank];

        for (int i = 0; i < tensor1.Rank; i++)
        {
            sizes[i] = tensor1.GetLength(i);
        }

        return CompareElementsRecursive(
            tensor1,
            tensor2,
            0,
            indices,
            sizes);
    }

    private static bool CompareElementsRecursive(Array tensor1, Array tensor2, int dimension, int[] indices, int[] sizes)
    {
        if (dimension == tensor1.Rank)
        {
            return tensor1.GetValue(indices)?.Equals(tensor2.GetValue(indices)) ?? false;
        }

        for (int i = 0; i < sizes[dimension]; i++)
        {
            indices[dimension] = i;

            if (!CompareElementsRecursive(
                    tensor1,
                    tensor2,
                    dimension + 1,
                    indices,
                    sizes))
            {
                return false;
            }
        }

        return true;
    }
}