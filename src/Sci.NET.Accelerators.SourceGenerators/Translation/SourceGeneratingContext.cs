// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Sci.NET.Accelerators.SourceGenerators.Translation.Builders;

namespace Sci.NET.Accelerators.SourceGenerators.Translation;

internal class SourceGeneratingContext
{
    public SourceGeneratingContext(MethodDeclarationSyntax methodDeclarationSyntax, SourceProductionContext context, Compilation methodBuilder, TranslatedMethodBuilder translatedMethodBuilder)
    {
        MethodDeclarationSyntax = methodDeclarationSyntax;
        Context = context;
        TranslatedMethodBuilder = translatedMethodBuilder;
        Compilation = methodBuilder;
    }

    public MethodDeclarationSyntax MethodDeclarationSyntax { get; }

    public SourceProductionContext Context { get; }

    public TranslatedMethodBuilder TranslatedMethodBuilder { get; }

    public Compilation Compilation { get; }
}