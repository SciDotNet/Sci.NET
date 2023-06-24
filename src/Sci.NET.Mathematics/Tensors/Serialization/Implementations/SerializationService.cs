// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using Sci.NET.Common.Attributes;
using Sci.NET.Common.Performance;

namespace Sci.NET.Mathematics.Tensors.Serialization.Implementations;

internal class SerializationService : ISerializationService
{
    [PreviewFeature]
    [MethodImpl(ImplementationOptions.AggressiveOptimization)]
    public unsafe void SaveNpy<TNumber>(ITensor<TNumber> tensor, string path)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var stream = File.OpenWrite(path);

        var header = new Span<byte>("\x93NUMPY\x01\x00"u8.ToArray(), 1, 8);
        var version = "v\x00"u8;
        var description = "{'descr': '"u8;
        var dType = GetNumpyDataType<TNumber>();
        var fortranOrder = "', 'fortran_order': False"u8;
        var shapeStart = ", 'shape': ("u8;
        var shapeBody = Encoding.UTF8.GetBytes(string.Join(", ", tensor.Shape));
        var shapeEnd = "), }"u8;

        var headerLength = header.Length +
                           version.Length +
                           description.Length +
                           dType.Length +
                           fortranOrder.Length +
                           shapeStart.Length +
                           shapeBody.Length +
                           shapeEnd.Length;

        // Pad the header to a multiple of 16 bytes
        var paddingLength = 0;

        if (headerLength % 16 != 0)
        {
            var n = (int)Math.Ceiling(headerLength / 16.0f);
            paddingLength = (16 * (n + 3)) - headerLength;
        }

        var paddingBytes = stackalloc byte[paddingLength];
        var paddingRef = new Span<byte>(paddingBytes, paddingLength);

        for (var i = 0; i < paddingLength - 1; i++)
        {
            paddingRef[i] = 0x20;
        }

        paddingRef[paddingLength - 1] = 0x0A;

        for (var i = 0; i < paddingLength - 1; i++)
        {
            paddingRef[i] = 0x20;
        }

        paddingRef[paddingLength - 1] = 0x0A;

        stream.Write(header);
        stream.Write(version);
        stream.Write(description);
        stream.Write(dType);
        stream.Write(fortranOrder);
        stream.Write(shapeStart);
        stream.Write(shapeBody);
        stream.Write(shapeEnd);
        stream.Write(paddingRef);

        tensor.Handle.WriteTo(stream);

        stream.Flush();
        stream.Close();
    }

    // ReSharper disable once CyclomaticComplexity -- This is a switch statement, it's supposed to be complex
    private static ReadOnlySpan<byte> GetNumpyDataType<TNumber>()
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TNumber.Zero switch
        {
            byte => "<u1"u8,
            sbyte => "<i1"u8,
            ushort => "<u2"u8,
            short => "<i2"u8,
            uint => "<u4"u8,
            int => "<i4"u8,
            ulong => "<u8"u8,
            long => "<i8"u8,
            float => "<f4"u8,
            double => "<f8"u8,
            _ => throw new NotSupportedException()
        };
    }
}