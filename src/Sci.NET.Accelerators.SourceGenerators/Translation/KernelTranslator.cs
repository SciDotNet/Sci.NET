// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Sci.NET.Accelerators.SourceGenerators.Diagnostics;
using Sci.NET.Accelerators.SourceGenerators.Extensions;
using Sci.NET.Accelerators.SourceGenerators.Translation.Builders;

namespace Sci.NET.Accelerators.SourceGenerators.Translation;

internal class KernelTranslator
{
    private readonly SourceGeneratingContext _context;

    public KernelTranslator(MethodDeclarationSyntax methodDeclarationSyntax, SourceProductionContext context, Compilation compilation)
    {
        _context = new SourceGeneratingContext(
            methodDeclarationSyntax,
            context,
            compilation,
            new TranslatedMethodBuilder());
    }

    public void Translate()
    {
        ValidateReturnType();
        AddParameters();
        ProcessMethodBody();
    }

    private void AddParameters()
    {
        foreach (var parameter in _context.MethodDeclarationSyntax.ParameterList.Parameters)
        {
            if (parameter.Modifiers.Any())
            {
                TranslationDiagnostics.ReportKernelParameterModifiersNotSupported(parameter, _context.Context);
            }
            else if (!parameter.Type?.IsSupportedNumberType() ?? false)
            {
                TranslationDiagnostics.ReportParameterTypeNotSupported(parameter.Type, _context.Context);
            }
            else
            {
                _context.TranslatedMethodBuilder.AddParameter(parameter);
            }
        }
    }

    private void ValidateReturnType()
    {
        if (_context.MethodDeclarationSyntax.ReturnType is not PredefinedTypeSyntax predefinedTypeSyntax)
        {
            TranslationDiagnostics.ReportReturnTypeNotSupported(_context.MethodDeclarationSyntax.ReturnType, _context.Context);
            return;
        }

        if (!predefinedTypeSyntax.Keyword.IsKind(SyntaxKind.VoidKeyword))
        {
            TranslationDiagnostics.ReportReturnTypeNotSupported(_context.MethodDeclarationSyntax.ReturnType, _context.Context);
        }
    }

    private void ProcessMethodBody()
    {
        if (_context.MethodDeclarationSyntax.ExpressionBody is not null)
        {
            TranslationDiagnostics.ReportKernelExpressionBodyNotSupported(_context.MethodDeclarationSyntax.ExpressionBody, _context.Context);
            return;
        }

        if (_context.MethodDeclarationSyntax.Body is null)
        {
            TranslationDiagnostics.ReportEmptyKernelBody(_context.MethodDeclarationSyntax, _context.Context);
        }
    }
}