// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.ComponentModel;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Sci.NET.Common.Runtime;

/// <summary>
/// A helper class for resolving native library paths.
/// </summary>
[PublicAPI]
public static class RuntimeDllImportResolver
{
    /// <summary>
    /// Loads a native library.
    /// </summary>
    /// <param name="libraryName">The library name to look for.</param>
    /// <param name="assembly">The assembly to load to.</param>
    /// <returns>A pointer to the loaded library.</returns>
    /// <exception cref="InvalidOperationException">The library failed to load.</exception>
    /// <exception cref="PlatformNotSupportedException">The current platform is not supported.</exception>
    public static nint LoadLibrary(string libraryName, Assembly assembly)
    {
        var assemblyDirectory = Path.GetDirectoryName(assembly.Location) ??
                                throw new InvalidOperationException("The assembly directory could not be resolved.");

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            if (TryLoadWindowsLibrary(libraryName, assemblyDirectory, out var handle))
            {
                return handle;
            }

            throw new InvalidOperationException("The native library could not be loaded.");
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            if (TryLoadLinuxLibrary(libraryName, assemblyDirectory, out var handle))
            {
                return handle;
            }

            throw new InvalidOperationException("The native library could not be loaded.");
        }

        throw new PlatformNotSupportedException("This platform is not supported.");
    }

    private static bool TryLoadLinuxLibrary(string libraryName, string assemblyDirectory, out nint o)
    {
        return TryResolve($@"{assemblyDirectory}\runtimes\linux-x64\{libraryName}.so", out o) ||
               TryResolve($@"{assemblyDirectory}\{libraryName}.so", out o) ||
               TryResolve($@"{assemblyDirectory}\{libraryName}.so.1", out o);
    }

    private static bool TryLoadWindowsLibrary(string libraryName, string assemblyDirectory, out nint o)
    {
        return TryResolve($@"{assemblyDirectory}\CUDA\win\x64\{libraryName}.dll", out o) ||
               TryResolve($@"{assemblyDirectory}\CUDA\win-x64\{libraryName}.dll", out o) ||
               TryResolve($@"{assemblyDirectory}\{libraryName}.dll", out o);
    }

    private static bool TryResolve(string libraryName, out nint handle)
    {
        handle = nint.Zero;
        var unsupportedPlatform = true;

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            handle = WinApi.LoadLibrary(libraryName);
            unsupportedPlatform = false;

            if (handle != nint.Zero)
            {
                return true;
            }
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            handle = LinuxApi.Dlopen(libraryName, LinuxApi.RtldNow);
            unsupportedPlatform = false;

            if (handle != nint.Zero)
            {
                return true;
            }
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            handle = DarwinApi.Dlopen(libraryName, DarwinApi.RtldNow);
            unsupportedPlatform = false;

            if (handle != nint.Zero)
            {
                return true;
            }
        }

        if (!unsupportedPlatform)
        {
            throw new InvalidOperationException(GetLastError());
        }

        throw new PlatformNotSupportedException("This platform is not supported.");
    }

    private static string GetLastError()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            var error = Marshal.GetLastWin32Error();
            return new Win32Exception(error).Message;
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            return Marshal.PtrToStringAnsi(LinuxApi.Dlerror()) ?? string.Empty;
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            return Marshal.PtrToStringAnsi(DarwinApi.Dlerror()) ?? string.Empty;
        }

        throw new PlatformNotSupportedException("This platform is not supported.");
    }

    private static class WinApi
    {
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        public static extern nint LoadLibrary(string dllName);
    }

    private static class LinuxApi
    {
        public const int RtldNow = 2;

        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [DllImport("libdl.so", EntryPoint = "dlopen", CharSet = CharSet.Unicode)]
        public static extern nint Dlopen(string fileName, int flags);

        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [DllImport("libdl.so", EntryPoint = "dlerror")]
        public static extern nint Dlerror();
    }

    private static class DarwinApi
    {
        public const int RtldNow = 2;

        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [DllImport("libSystem.dylib", EntryPoint = "dlopen", CharSet = CharSet.Unicode)]
        public static extern nint Dlopen(string fileName, int flags);

        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        [DllImport("libSystem.dylib", EntryPoint = "dlerror")]
        public static extern nint Dlerror();
    }
}
#pragma warning restore CA1060