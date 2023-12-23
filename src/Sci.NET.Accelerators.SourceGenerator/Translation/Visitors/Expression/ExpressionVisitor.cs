// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Sci.NET.Accelerators.SourceGenerator.Translation.Visitors.Expression;

internal class ExpressionVisitor : ITranslationVisitor<ExpressionSyntax>
{
    public ExpressionSyntax Visit(ExpressionSyntax syntaxNode, TranslationContext context)
    {
        return syntaxNode;
    }
}