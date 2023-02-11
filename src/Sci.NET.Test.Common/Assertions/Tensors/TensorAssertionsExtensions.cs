// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Test.Common.Assertions.Tensors;

/// <summary>
/// Extension methods for <see cref="ITensor{TCollection}"/> assertions.
/// </summary>
[PublicAPI]
[ExcludeFromCodeCoverage]
[DebuggerNonUserCode]
public static class TensorAssertionsExtensions
{
    /// <summary>
    /// Assertions for <see cref="ITensor{TCollection}"/>.
    /// </summary>
    /// <param name="tensor">The target memory block.</param>
    /// <typeparam name="TCollection">The type of element in the collection.</typeparam>
    /// <returns>A <see cref="ITensor{TCollection}"/> instance.</returns>
    public static TensorAssertions<TCollection> Should<TCollection>(this ITensor<TCollection> tensor)
        where TCollection : unmanaged, INumber<TCollection>
    {
        return new TensorAssertions<TCollection>(tensor);
    }
}