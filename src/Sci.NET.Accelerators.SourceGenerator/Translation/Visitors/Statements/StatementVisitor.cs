// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Sci.NET.Accelerators.SourceGenerator.Translation.Visitors.Statements;

internal class StatementVisitor : ITranslationVisitor<StatementSyntax>
{
    public StatementSyntax Visit(StatementSyntax syntaxNode, TranslationContext context)
    {
        return syntaxNode switch
        {
            BlockSyntax blockSyntax => new BlockSyntaxVisitor().Visit(blockSyntax, context),
            ForStatementSyntax forStatementSyntax => new ForStatementSyntaxVisitor().Visit(forStatementSyntax, context),
            ExpressionStatementSyntax expressionStatementSyntax => new ExpressionStatementSyntaxVisitor().Visit(expressionStatementSyntax, context),
            IfStatementSyntax ifStatementSyntax => new IfStatementSyntaxVisitor().Visit(ifStatementSyntax, context),
            _ => syntaxNode
        };
    }
}