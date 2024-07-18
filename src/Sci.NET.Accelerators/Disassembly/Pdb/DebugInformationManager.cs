// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection;

namespace Sci.NET.Accelerators.Disassembly.Pdb;

internal sealed class DebugInformationManager : IDisposable
{
    private static readonly DebugInformationManager _instance = new();

    private readonly ReaderWriterLockSlim _lock;
    private readonly Dictionary<Assembly, IAssemblyDebugInformation> _assemblyDebugInformation;

    private DebugInformationManager()
    {
        _lock = new ReaderWriterLockSlim();
        _assemblyDebugInformation = new Dictionary<Assembly, IAssemblyDebugInformation>();
    }

    public static bool TryLoadMethodDebugInformation(MethodBase method, out MethodDebugInfo methodDebugInfo)
    {
        _instance._lock.EnterUpgradeableReadLock();

        try
        {
            if (_instance._assemblyDebugInformation.TryGetValue(method.DeclaringType!.Assembly, out var assemblyDebugInformation) && assemblyDebugInformation.TryGetMethodDebugInfo(method, out var methodInfo))
            {
                methodDebugInfo = methodInfo;
                return true;
            }

            _instance._lock.EnterWriteLock();

            try
            {
                var pdbFile = method.DeclaringType.Assembly.Location.Replace(".dll", ".pdb", StringComparison.Ordinal);

                using var pdbStream = File.OpenRead(pdbFile);
                assemblyDebugInformation = new AssemblyDebugInformation(method.DeclaringType.Assembly, pdbStream!);
                _instance._assemblyDebugInformation.Add(method.DeclaringType.Assembly, assemblyDebugInformation);

                if (!assemblyDebugInformation.TryGetMethodDebugInfo(method, out methodInfo))
                {
                    throw new InvalidOperationException("Method debug information not loaded.");
                }

                methodDebugInfo = methodInfo;
            }
            catch
            {
                var fakeAssemblyDebugInformation = new FakeAssemblyDebugInformation(method.DeclaringType.Assembly);
                _instance._assemblyDebugInformation.Add(method.DeclaringType.Assembly, fakeAssemblyDebugInformation);

                if (!fakeAssemblyDebugInformation.TryGetMethodDebugInfo(method, out var fakeMethodInfo))
                {
                    throw new InvalidOperationException("Method debug information not loaded.");
                }

                methodDebugInfo = fakeMethodInfo;
            }
            finally
            {
                _instance._lock.ExitWriteLock();
            }
        }
        finally
        {
            _instance._lock.ExitUpgradeableReadLock();
        }

        return true;
    }

    public void Dispose()
    {
        _lock.Dispose();
    }
}