// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.Rewriter;

namespace Sci.NET.Accelerators.CodeGeneration.IL;

internal class IlCompiler : IDeviceCompiler<IlKernel>
{
    static IlCompiler()
    {
        Identifier = Guid.NewGuid();
    }

    public static Guid Identifier { get; }

    public IlKernel Compile(MsilSsaMethod intermediateRepresentation)
    {
        return new ("Kernel");
    }
}