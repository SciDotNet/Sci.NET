// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Sci.NET.Accelerators.SourceGenerators.Translation.Builders;

internal class TranslatedMethodBuilder
{
    public List<ParameterSyntax> Parameters { get; } = new ();

    public void AddParameter(ParameterSyntax parameter)
    {
        Parameters.Add(parameter);
    }
}