// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.Memory.ReferenceCounting;

/// <summary>
/// An interface for a derived type which is reference counted.
/// </summary>
[PublicAPI]
public interface IReferenceCounted
{
    /// <summary>
    /// Gets number of references to the given handle.
    /// </summary>
    public ReferenceCount ReferenceCount { get; }
}