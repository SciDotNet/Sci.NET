// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Mathematics.Backends;
using Sci.NET.Mathematics.Backends.Devices;
using Sci.NET.Mathematics.Tensors.Exceptions;

namespace Sci.NET.Mathematics.Tensors.Common.Implementations;

internal class DeviceGuardService : IDeviceGuardService
{
    public ITensorBackend GuardBinaryOperation(IDevice left, IDevice right)
    {
        if (!left.Equals(right))
        {
            throw new TensorDataLocalityException(
                "The left and right operands must be on the same device, but were on {0} and {1}'",
                left,
                right);
        }

        return left.GetTensorBackend();
    }

    public ITensorBackend GuardMultiParameterOperation(params IDevice[] devices)
    {
        var allEqual = devices.DistinctBy(x => x.Id).Count() == 1;

        if (!allEqual)
        {
            throw new TensorDataLocalityException("All operands must be on the same device, but were on {0}", devices);
        }

        return devices[0].GetTensorBackend();
    }
}