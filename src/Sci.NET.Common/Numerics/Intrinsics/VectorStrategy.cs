// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.Numerics.Intrinsics;

/// <summary>
/// An enumeration to determine if an operation should be vectorized or not.
/// </summary>
[PublicAPI]
public enum VectorStrategy
{
    /// <summary>
    /// None.
    /// </summary>
    None = 0,

    /// <summary>
    /// Should be loaded as a vector.
    /// </summary>
    Vector = 1,

    /// <summary>
    /// Should be loaded as a scalar.
    /// </summary>
    Scalar = 2
}