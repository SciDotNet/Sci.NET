// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Common.Memory;

/// <summary>
/// Represents a reference counted object.
/// </summary>
[PublicAPI]
public interface IReferenceCounted
{
    /// <summary>
    /// Adds a reference to the object.
    /// </summary>
    /// <param name="id">The guid of the reference to add.</param>
    public void Rent(Guid id);

    /// <summary>
    /// Releases a reference to the object.
    /// </summary>
    /// <param name="id">The guid of the reference to release.</param>
    public void Release(Guid id);
}