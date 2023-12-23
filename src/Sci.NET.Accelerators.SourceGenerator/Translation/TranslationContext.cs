// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Sci.NET.Accelerators.SourceGenerator.Translation;

internal class TranslationContext(MethodDeclarationSyntax methodDeclarationSyntax, SourceProductionContext sourceProductionContext, Compilation compilation)
{
    public MethodDeclarationSyntax MethodDeclarationSyntax { get; } = methodDeclarationSyntax;

    public SourceProductionContext SourceProductionContext { get; } = sourceProductionContext;

    public Compilation Compilation { get; } = compilation;

    public ParameterListSyntax ParameterListSyntax { get; set; }

    public BlockSyntax EntryBlockSyntax { get; set; }
}