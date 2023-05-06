// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Numerics;
using Sci.NET.Common.Memory;
using Sci.NET.Mathematics.Backends;

namespace Sci.NET.Mathematics.Tensors;

/// <summary>
/// Represents a matrix.
/// </summary>
/// <typeparam name="TNumber">The number type of the <see cref="Matrix{TNumber}"/>.</typeparam>
[PublicAPI]
[DebuggerDisplay("{ToArray()}")]
public class Matrix<TNumber> : Tensor<TNumber>
    where TNumber : unmanaged, INumber<TNumber>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Matrix{TNumber}"/> class.
    /// </summary>
    /// <param name="rows">The number of rows in the matrix.</param>
    /// <param name="columns">The number of columns in the matrix.</param>
    /// <param name="backend">The backend to use.</param>
    public Matrix(int rows, int columns, ITensorBackend? backend = null)
        : base(backend, rows, columns)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Matrix{TNumber}"/> class.
    /// </summary>
    /// <param name="rows">The number of rows in the matrix.</param>
    /// <param name="columns">The number of columns in the matrix.</param>
    /// <param name="memoryBlock">The memory block for the matrix.</param>
    /// <param name="backend">The backend type for the <see cref="Matrix{TNumber}"/>.</param>
    public Matrix(int rows, int columns, IMemoryBlock<TNumber> memoryBlock, ITensorBackend? backend = null)
        : base(memoryBlock, new Shape(rows, columns), backend ?? Tensor.DefaultBackend)
    {
    }

    /// <summary>
    /// Gets the number of rows in the matrix.
    /// </summary>
    public int Rows => Shape[0];

    /// <summary>
    /// Gets the number of columns in the matrix.
    /// </summary>
    public int Columns => Shape[1];

    /// <summary>
    /// Gets the debugger display object.
    /// </summary>
    [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
    private protected Array DebuggerDisplayObject => ToArray();
}