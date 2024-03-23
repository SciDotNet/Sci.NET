// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection;
using Sci.NET.Accelerators.Disassembly.Pdb;

namespace Sci.NET.Accelerators.UnitTests.Disassembler;

public class AssemblyDebugInformationShould
{
    [Fact]
    public void Ctor_LoadsModulesAndMethods()
    {
        var assembly = Assembly.GetExecutingAssembly();
        var pdbFile = assembly.Location.Replace(".dll", ".pdb", StringComparison.Ordinal);

        using var pdbStream = File.OpenRead(pdbFile);
        var assemblyDebugInformation = new AssemblyDebugInformation(assembly, pdbStream!);

        assemblyDebugInformation.Assembly.Should().NotBeNull();

        var methodBase = GetType().GetMethod(nameof(Ctor_LoadsModulesAndMethods), BindingFlags.Instance | BindingFlags.Public) ?? throw new InvalidOperationException();

        assemblyDebugInformation.TryGetMethodDebugInfo(methodBase, out var methodDebugInfo).Should().BeTrue();
    }
}