// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// An interface for a rank-N tensor, which is an immutable N-Dimensional array.
/// </summary>
/// <typeparam name="TNumber">The type of the numbers stored in the <see cref="ITensor{TNumber}"/>.</typeparam>
[PublicAPI]
public interface ITensor<TNumber> : ITensorLocalityOperations
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Gets the shape of the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    public Shape Shape { get; }

    /// <summary>
    /// Gets the memory of the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    public IMemoryBlock<TNumber> Memory { get; }

    /// <summary>
    /// Gets the backend of the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    public ITensorBackend Backend { get; }

    /// <summary>
    /// Gets a value indicating whether the <see cref="ITensor{TNumber}"/> owns the memory it points to.
    /// </summary>
    public bool IsMemoryOwner { get; }

    /// <summary>
    /// Gets the gradient of the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    [MemberNotNullWhen(true, nameof(RequiresGradient))]
    public ITensor<TNumber>? Gradient { get; }

    /// <summary>
    /// Gets a value indicating whether the <see cref="ITensor{TNumber}"/> requires a gradient.
    /// </summary>
    public bool RequiresGradient { get; }

    /// <summary>
    /// Gets a value indicating whether the <see cref="ITensor{TNumber}"/> is a gradient.
    /// </summary>
    public bool IsGradient { get; }

    /// <summary>
    /// Gets the parent nodes of the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    protected internal ICollection<(string Name, ITensor<TNumber> Parent, Func<ITensor<TNumber>, ITensor<TNumber>> Gradient)> Parents { get; }

    /// <summary>
    /// Gets the slice of the <see cref="ITensor{TNumber}"/> at the specified indices.
    /// </summary>
    /// <param name="indices">The indices of the <see cref="ITensor{TNumber}"/> to slice.</param>
#pragma warning disable CA1043
    public ITensor<TNumber> this[params int[] indices] { get; }
#pragma warning restore CA1043

    /// <summary>
    /// Propagates the gradient backward through the computation graph.
    /// </summary>
    public void Backward();

    /// <summary>
    /// Adds a parent node to the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <param name="name">The name of the backwards function.</param>
    /// <param name="parent">The parent node to add.</param>
    /// <param name="gradientFunc">The gradient function to apply to the parent node.</param>
    public void AddParent(string name, ITensor<TNumber> parent, Func<ITensor<TNumber>, ITensor<TNumber>> gradientFunc)
    {
        Parents.Add((name, parent, gradientFunc));
    }

    /// <summary>
    /// Creates an instance of the <see cref="ITensor{TNumber}"/> with the gradient.
    /// </summary>
    /// <returns>The recreated <see cref="ITensor{TNumber}"/>.</returns>
    public ITensor<TNumber> WithGradient()
    {
        return new Tensor<TNumber>(Memory, Shape, Backend, requiresGradient: true);
    }

    /// <summary>
    /// Recreates the <see cref="ITensor{TNumber}"/> as a gradient.
    /// </summary>
    /// <returns>The recreated <see cref="ITensor{TNumber}"/> as a gradient.</returns>
    public ITensor<TNumber> AsGradient()
    {
        var value = new Tensor<TNumber>(Memory, Shape, Backend, requiresGradient: false) { IsGradient = true };

        if (RequiresGradient)
        {
            value.Gradient?.Parents.Clear();
            value.Gradient?.Dispose();
        }

        return value;
    }

    /// <summary>
    /// Copies the values of the <see cref="ITensor{TNumber}"/> to an <see cref="Array"/>.
    /// </summary>
    /// <returns>An array with the same data as the <see cref="ITensor{TNumber}"/>.</returns>
    /// <exception cref="InvalidOperationException">The operation could not be completed.</exception>
    public Array ToArray();

    /// <summary>
    /// Creates an instance of <see cref="Vector{TNumber}"/> from the <see cref="ITensor{TNumber}"/>,
    /// assuming that the <see cref="ITensor{TNumber}"/> is a scalar.
    /// </summary>
    /// <returns>The <see cref="ITensor{TNumber}"/> instance as a <see cref="Vector{TNumber}"/>.</returns>
    /// <exception cref="InvalidShapeException">Throws when the shape of the <see cref="ITensor{TNumber}"/> is invalid.</exception>
    public Scalar<TNumber> ToScalar()
    {
        if (!Shape.IsScalar)
        {
            throw new InvalidShapeException($"The tensor must be a scalar, but got shape {Shape}");
        }

        DetachMemory();

        var result = new Scalar<TNumber>(Memory, Backend, RequiresGradient);

        foreach (var parent in Parents)
        {
            ((ITensor<TNumber>)result).AddParent(parent.Name, parent.Parent, parent.Gradient);
        }

        return result;
    }

    /// <summary>
    /// Creates an instance of <see cref="Vector{TNumber}"/> from the <see cref="ITensor{TNumber}"/>,
    /// assuming that the <see cref="ITensor{TNumber}"/> is 1-dimensional.
    /// </summary>
    /// <returns>The <see cref="ITensor{TNumber}"/> instance as a <see cref="Vector{TNumber}"/>.</returns>
    /// <exception cref="InvalidShapeException">Throws when the shape of the <see cref="ITensor{TNumber}"/> is invalid.</exception>
    public Vector<TNumber> ToVector()
    {
        if (!Shape.IsVector)
        {
            throw new InvalidShapeException(
                $"The tensor must be 1-dimensional to be converted to a vector, but got shape {Shape}");
        }

        DetachMemory();

        var result = new Vector<TNumber>(Shape.Dimensions[0], Memory, Backend, RequiresGradient);

        foreach (var parent in Parents)
        {
            ((ITensor<TNumber>)result).AddParent(parent.Name, parent.Parent, parent.Gradient);
        }

        return result;
    }

    /// <summary>
    /// Creates an instance of <see cref="Matrix{TNumber}"/> from the <see cref="ITensor{TNumber}"/>,
    /// assuming that the <see cref="ITensor{TNumber}"/> is 2-dimensional.
    /// </summary>
    /// <param name="requiresGradient">A value indicating whether the resulting tensor requires a gradient.</param>
    /// <returns>The <see cref="ITensor{TNumber}"/> instance as a <see cref="Matrix{TNumber}"/>.</returns>
    /// <exception cref="InvalidShapeException">Throws when the shape of the <see cref="ITensor{TNumber}"/> is invalid.</exception>
    public Matrix<TNumber> ToMatrix(bool? requiresGradient = null)
    {
        if (!Shape.IsMatrix)
        {
            throw new InvalidShapeException(
                $"The tensor must be 2-dimensional to be converted to a matrix, but got shape {Shape}");
        }

        DetachMemory();

        var result = new Matrix<TNumber>(Shape.Dimensions[0], Shape.Dimensions[1], Memory, Backend, requiresGradient ?? RequiresGradient);

        foreach (var parent in Parents)
        {
            ((ITensor<TNumber>)result).AddParent(parent.Name, parent.Parent, parent.Gradient);
        }

        return result;
    }

    /// <summary>
    /// Creates an instance of <see cref="Tensor{TNumber}"/> from the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    /// <returns>The <see cref="ITensor{TNumber}"/> instance as a <see cref="Tensor{TNumber}"/>.</returns>
    public Tensor<TNumber> ToTensor()
    {
        DetachMemory();

        var result = new Tensor<TNumber>(this, Shape);

        foreach (var parent in Parents)
        {
            ((ITensor<TNumber>)result).AddParent(parent.Name, parent.Parent, parent.Gradient);
        }

        return result;
    }

    /// <summary>
    /// Checks if the <see cref="ITensor{TNumber}"/> is a <see cref="Scalar{TNumber}"/>.
    /// </summary>
    /// <returns><c>true</c> if the <see cref="ITensor{TNumber}"/> is a <see cref="Scalar{TNumber}"/> else, <c>false</c>.</returns>
    public bool IsScalar()
    {
        return Shape.IsScalar;
    }

    /// <summary>
    /// Checks if the <see cref="ITensor{TNumber}"/> is a <see cref="Vector{TNumber}"/>.
    /// </summary>
    /// <returns><c>true</c> if the <see cref="ITensor{TNumber}"/> is a <see cref="Vector{TNumber}"/> else, <c>false</c>.</returns>
    public bool IsVector()
    {
        return Shape.IsVector;
    }

    /// <summary>
    /// Checks if the <see cref="ITensor{TNumber}"/> is a <see cref="Matrix{TNumber}"/>.
    /// </summary>
    /// <returns><c>true</c> if the <see cref="ITensor{TNumber}"/> is a <see cref="Matrix{TNumber}"/> else, <c>false</c>.</returns>
    public bool IsMatrix()
    {
        return Shape.IsMatrix;
    }

    /// <summary>
    /// Checks if the <see cref="ITensor{TNumber}"/> is a <see cref="Tensor{TNumber}"/>.
    /// </summary>
    /// <returns><c>true</c> if the <see cref="ITensor{TNumber}"/> is a <see cref="Tensor{TNumber}"/> else, <c>false</c>.</returns>
    public bool IsTensor()
    {
        return Shape.IsTensor;
    }

    /// <summary>
    /// Forces the disposal of the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    public void ForceDispose();

    /// <summary>
    /// Detaches the memory from the <see cref="ITensor{TNumber}"/>.
    /// </summary>
    protected internal void DetachMemory();

    /// <summary>
    /// Propagates the gradient backward through the computation graph.
    /// </summary>
    /// <exception cref="InvalidShapeException">Throws when the shape of the parent gradient tensor is invalid.</exception>
    protected internal void BackwardInternal()
    {
        // First, propagate to each parent
        foreach (var (_, parent, gradientFunc) in Parents)
        {
            if (parent.RequiresGradient)
            {
                var parentGradient = gradientFunc(Gradient!);

                parent.AccumulateGradient(parentGradient);
            }
        }

        // Ensure each parent node gets its accumulated gradient and continues to propagate
        foreach (var (_, parent, _) in Parents)
        {
            parent.BackwardInternal();
        }
    }

    /// <summary>
    /// Accumulates the gradient of the parent node.
    /// </summary>
    /// <param name="parentGradient">The gradient of the parent node.</param>
    /// <exception cref="InvalidShapeException">Throws when the shape of the parent gradient tensor is invalid.</exception>
    protected void AccumulateGradient(ITensor<TNumber> parentGradient)
    {
        ArgumentNullException.ThrowIfNull(Gradient);
        InvalidShapeException.ThrowIfDifferentShape(Gradient.Shape, parentGradient.Shape);

        if (parentGradient.Shape != Shape)
        {
            throw new InvalidShapeException(
                $"The shape of the parent gradient tensor must be the same as the shape of the tensor, but got {parentGradient.Shape} and {Shape}");
        }

        if (RequiresGradient)
        {
            using var newTensor = TensorServiceProvider
                .GetTensorOperationServiceProvider()
                .GetArithmeticService()
                .Add(Gradient, parentGradient);

            newTensor.Memory.CopyTo(Gradient.Memory);
        }
    }
}