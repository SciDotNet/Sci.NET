// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Reflection;
using Sci.NET.Accelerators.IR;

namespace Sci.NET.Accelerators.Extensions;

/// <summary>
/// Extensions for the <see cref="Type"/> class.
/// </summary>
[PublicAPI]
public static class TypeExtensions
{
    /// <summary>
    /// Converts a <see cref="Type"/> to an <see cref="IrType"/>.
    /// </summary>
    /// <param name="type">The <see cref="Type"/> to convert.</param>
    /// <returns>The <see cref="Type"/> converted to an <see cref="IrType"/>.</returns>
    /// /// <exception cref="NotSupportedException">
    /// Thrown in the following cases:
    /// <list type="number">
    ///   <item>
    ///     <description>The type is a byref or pointer type without an element type.</description>
    ///   </item>
    ///   <item>
    ///     <description>The type is a generic type.</description>
    ///   </item>
    ///   <item>
    ///     <description>The type is an unsupported primitive type.</description>
    ///   </item>
    ///   <item>
    ///     <description>The type falls into any other unsupported category.</description>
    ///   </item>
    /// </list>
    /// </exception>
    public static IrType ToIrType(this Type type)
    {
        if (type.IsByRef)
        {
            return IrType.MakePointer(ToIrType(type.GetElementType() ?? throw new NotSupportedException("ByRef type has no element type.")));
        }

        if (type.IsPointer)
        {
            return IrType.MakePointer(ToIrType(type.GetElementType() ?? throw new NotSupportedException("Pointer type has no element type.")));
        }

        if (type.IsInterface || type.IsAbstract || type.IsClass)
        {
            return IrType.MakePointer(IrType.Void);
        }

        if (type.IsGenericType)
        {
            throw new NotSupportedException("Generic types are not supported.");
        }

        if (type.IsPrimitive)
        {
            return MakePrimitiveType(type);
        }

        if (type == typeof(void))
        {
            return IrType.Void;
        }

        if (type.IsValueType && type is { IsPrimitive: false, IsEnum: false } && type != typeof(ValueType))
        {
            return MakeStructureType(type);
        }

        throw new NotSupportedException("Unsupported type.");
    }

    private static IrType MakePrimitiveType(Type type)
    {
        return type switch
        {
            _ when type == typeof(nint) || type == typeof(nuint) => IrType.NativeInt,
            _ when type == typeof(bool) => IrType.Boolean,
            _ when type == typeof(char) => IrType.Char,
            _ when type == typeof(string) => IrType.String,
            _ when type == typeof(byte) || type == typeof(sbyte) => IrType.Int8,
            _ when type == typeof(short) || type == typeof(ushort) => IrType.Int16,
            _ when type == typeof(int) || type == typeof(uint) => IrType.Int32,
            _ when type == typeof(long) || type == typeof(ulong) => IrType.Int64,
            _ when type == typeof(Half) => IrType.Fp16,
            _ when type == typeof(float) => IrType.Fp32,
            _ when type == typeof(double) => IrType.Fp64,
            _ => throw new NotSupportedException("Unsupported primitive type.")
        };
    }

    private static IrType MakeStructureType(Type type)
    {
        var fields = type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        var fieldTypes = fields.Select(x => x.FieldType.ToIrType()).ToList();
        var totalSize = fieldTypes.Sum(x => x.Bits);
        var name = $"type {{ {string.Join(", ", fieldTypes.Select(x => x.ToString()))} }}";

        return new IrType { Name = name, Bits = totalSize, IsIntrinsic = false, DotnetType = type, IsPointer = false };
    }
}