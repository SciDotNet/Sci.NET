// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Sci.NET.Accelerators.SourceGenerator.Translation.Visitors.Expression;

namespace Sci.NET.Accelerators.SourceGenerator.Translation.Visitors.Statements;

internal class IfStatementSyntaxVisitor : ITranslationVisitor<IfStatementSyntax>
{
    public IfStatementSyntax Visit(IfStatementSyntax syntaxNode, TranslationContext context)
    {
        var condition = new ExpressionVisitor().Visit(syntaxNode.Condition, context);
        var ifBlock = new StatementVisitor().Visit(syntaxNode.Statement, context);
        var elseBlock = syntaxNode.Else is not null ? new StatementVisitor().Visit(syntaxNode.Else.Statement, context) : null;

        return elseBlock is null ? SyntaxFactory.IfStatement(condition, ifBlock) : SyntaxFactory.IfStatement(condition, ifBlock, SyntaxFactory.ElseClause(elseBlock));
    }
}