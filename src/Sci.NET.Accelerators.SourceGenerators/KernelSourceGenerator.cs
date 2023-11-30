// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Sci.NET.Accelerators.SourceGenerators.Extensions;
using Sci.NET.Accelerators.SourceGenerators.Translation;

namespace Sci.NET.Accelerators.SourceGenerators;

/// <summary>
/// A sample source generator that creates a custom report based on class properties. The target class should be annotated with the 'Generators.ReportAttribute' attribute.
/// When using the source code as a baseline, an incremental source generator is preferable because it reduces the performance overhead.
/// </summary>
[Generator]
public class KernelSourceGenerator : IIncrementalGenerator
{
    private const string KernelAttributeName = "KernelAttribute";

    /// <summary>
    /// Initialize the generator.
    /// </summary>
    /// <param name="context">Initialization context.</param>
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var provider = context
            .SyntaxProvider
            .CreateSyntaxProvider(
                (s, _) => s is MethodDeclarationSyntax,
                (ctx, _) => GetMethodDeclarationForSourceGen(ctx))
            .Where(t => t.ReportAttributeFound)
            .Select((t, _) => t.Method);

        context.RegisterSourceOutput(
            context.CompilationProvider.Combine(provider.Collect()),
            (ctx, method) => GenerateCode(ctx, method.Left, method.Right));
    }

    /// <summary>
    /// Checks whether the Node is annotated with the [Kernel] attribute and maps syntax context to the specific node type (MethoDeclarationSyntax).
    /// </summary>
    /// <param name="context">Syntax context, based on CreateSyntaxProvider predicate.</param>
    /// <returns>The specific cast and whether the attribute was found.</returns>
    private static (MethodDeclarationSyntax Method, bool ReportAttributeFound) GetMethodDeclarationForSourceGen(GeneratorSyntaxContext context)
    {
        var methodDeclarationSyntax = (MethodDeclarationSyntax)context.Node;

        return (methodDeclarationSyntax, methodDeclarationSyntax.HasAttribute(KernelAttributeName));
    }

    /// <summary>
    /// Generate code action.
    /// </summary>
    /// <param name="context">Source generation context used to add source files.</param>
    /// <param name="compilation">Compilation used to resolve types.</param>
    /// <param name="methodDeclarations">Nodes annotated with the [Report] attribute that trigger the generate action.</param>
    private static void GenerateCode(
        SourceProductionContext context,
        Compilation compilation,
        ImmutableArray<MethodDeclarationSyntax> methodDeclarations)
    {
        foreach (var methodDeclarationSyntax in methodDeclarations)
        {
            var translator = new KernelTranslator(methodDeclarationSyntax, context, compilation);
            translator.Translate();
        }
    }
}