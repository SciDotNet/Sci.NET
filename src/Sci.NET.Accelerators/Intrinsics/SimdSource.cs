// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;

namespace Sci.NET.Accelerators.Intrinsics;

/// <summary>
/// A data source for SIMD instructions.
/// </summary>
/// <typeparam name="T">The type of data to be used.</typeparam>
[PublicAPI]
[SuppressMessage("Design", "CA1065:Do not raise exceptions in unexpected locations", Justification = "This is a marker type.")]
[SuppressMessage("Design", "CA1000:Do not declare static members on generic types", Justification = "This is a marker type.")]
[SuppressMessage("Roslynator", "RCS1158:Static member in generic type should use a type parameter", Justification = "This is a marker type.")]
public readonly ref struct SimdSource<T>
    where T : unmanaged, INumber<T>
{
    /// <summary>
    /// Gets the number of elements in the source.
    /// </summary>
    /// <exception cref="UnreachableException">Thrown if the property is accessed.</exception>
    public static int Count => throw new UnreachableException();

    /// <summary>
    /// Loads a value from the source.
    /// </summary>
    /// <param name="value">The pointer to the value to load.</param>
    /// <returns>The loaded value.</returns>
    /// <exception cref="UnreachableException">Thrown if the method is accessed.</exception>
    public static unsafe SimdSource<T> LoadFrom(T* value)
    {
        throw new UnreachableException();
    }
}