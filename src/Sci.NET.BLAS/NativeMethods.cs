// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Runtime.InteropServices;
using Sci.NET.BLAS.Types;

namespace Sci.NET.BLAS;

[SuppressMessage(
    "Security",
    "CA5392:Use DefaultDllImportSearchPaths attribute for P/Invokes",
    Justification = "The runtime will be packaged with the library.")]
internal static class NativeMethods
{
    private const string DllName = "SciNET.BLAS.Native.CPU";

    static NativeMethods()
    {
#pragma warning disable RCS1207
        NativeLibrary.SetDllImportResolver(typeof(NativeMethods).Assembly, DllImportResolver);
#pragma warning restore RCS1207
    }

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "allocate")]
    public static extern void AllocateMemory(long size, NumberType type, ref nint pointer);

    private static nint DllImportResolver(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != DllName)
        {
            return nint.Zero;
        }

        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new PlatformNotSupportedException("Only Windows is supported.");
        }

        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
        {
            throw new PlatformNotSupportedException("Only x64 is supported.");
        }

        var path = Path.Combine(
            Path.GetDirectoryName(assembly.Location)!,
            "runtimes",
            "win",
            $"{libraryName}.dll");

        return NativeLibrary.Load(path);
    }
}