// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

namespace Sci.NET.Accelerators.CodeGeneration.IL;

internal class IlKernel : ICompiledKernel
{
    public IlKernel(string name)
    {
        Name = name;
    }

    public string Name { get; }
}