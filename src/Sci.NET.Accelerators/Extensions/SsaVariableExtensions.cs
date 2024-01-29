// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Sci.NET.Accelerators.IR;
using Sci.NET.Accelerators.Rewriter.Variables;

namespace Sci.NET.Accelerators.Extensions;

internal static class SsaVariableExtensions
{
    public static IrValue ToIrValue<T>(this T variable)
        where T : ISsaVariable
    {
        return new () { Identifier = variable.Name, Type = variable.Type.ToIrType() };
    }
}