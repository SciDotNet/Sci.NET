// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Sci.NET.Mathematics.Backends.Devices;

namespace Sci.NET.Mathematics.Tensors.Exceptions;

/// <summary>
/// An exception thrown when a tensor operation is attempted on between <see cref="ITensor{TNumber}"/>
/// instances which are not stored on the same <see cref="IDevice"/>.
/// </summary>
[PublicAPI]
[ExcludeFromCodeCoverage]
public class TensorDataLocalityException : Exception
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TensorDataLocalityException"/> class.
    /// </summary>
    /// <param name="reason">The reason the devices are invalid.</param>
    /// <param name="devices">The devices of the <see cref="ITensor{TNumber}"/> associated with the operation.</param>
    [StringFormatMethod(nameof(reason))]
    public TensorDataLocalityException(string reason, params IDevice[] devices)
#pragma warning disable CA1305
        : base($"The devices are not compatible to operate between. {string.Format(reason, devices.ToList())}.")
#pragma warning restore CA1305
    {
        Devices = devices;
    }

    /// <summary>
    /// Gets the given shape.
    /// </summary>
    public IEnumerable<IDevice> Devices { get; }
}