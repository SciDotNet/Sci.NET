// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors;

namespace Sci.NET.Mathematics.Backends;

/// <summary>
/// An interface for <see cref="ITensor{TNumber}"/> locality operations.
/// </summary>
[PublicAPI]
public interface ITensorLocalityOperations
{
    /// <summary>
    /// Moves the tensor to the given device.
    /// </summary>
    /// <typeparam name="TDevice">The device to move to.</typeparam>
    public void To<TDevice>()
        where TDevice : IDevice, new();
}