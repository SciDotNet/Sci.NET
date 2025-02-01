// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.Common;

/// <summary>
/// An interface providing methods which guard against
/// operations taking place between <see cref="ITensor{TNumber}"/> instances
/// with backing memory stored on different devices.
/// </summary>
[PublicAPI]
public interface IDeviceGuardService
{
    /// <summary>
    /// Determines if both operands of a binary operation are stored on the
    /// same device and throws an exception if they are not.
    /// </summary>
    /// <param name="left">The left operand device.</param>
    /// <param name="right">The right operand device.</param>
    /// <exception cref="TensorDataLocalityException">Throws when the devices are not compatible.</exception>
    /// <returns>The backend used by the <paramref name="left"/> and <paramref name="right"/> devices.</returns>
    public ITensorBackend GuardBinaryOperation(IDevice left, IDevice right);

    /// <summary>
    /// Determines if all operands of a multi-parameter operation are stored on the
    /// same device, and throws an exception if they are not.
    /// </summary>
    /// <param name="devices">The devices for each parameter.</param>
    /// <exception cref="TensorDataLocalityException">Throws when the devices are not compatible.</exception>
    /// <returns>The backend used by the <paramref name="devices"/>.</returns>
    public ITensorBackend GuardMultiParameterOperation(params IDevice[] devices);
}