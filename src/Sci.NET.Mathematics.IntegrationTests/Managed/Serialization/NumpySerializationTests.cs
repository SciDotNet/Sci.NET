// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Mathematics.IntegrationTests.Managed.Serialization;

public class NumpySerializationTests
{
    private readonly string _resultsPath;

    public NumpySerializationTests()
    {
        _resultsPath = Path.Combine(Directory.GetCurrentDirectory(), "results\\managed\\serialization\\");

        if (!Directory.Exists(_resultsPath))
        {
            Directory.CreateDirectory(_resultsPath);
        }
    }
}