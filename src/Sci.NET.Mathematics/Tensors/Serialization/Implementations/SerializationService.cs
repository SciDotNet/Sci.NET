// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.IO.Compression;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using Newtonsoft.Json;
using Sci.NET.Common.Attributes;
using Sci.NET.Common.Exceptions;
using Sci.NET.Common.Numerics;
using Sci.NET.Common.Performance;
using Sci.NET.Common.Streams;
using Sci.NET.Mathematics.Tensors.Serialization.Implementations.Safetensors;

namespace Sci.NET.Mathematics.Tensors.Serialization.Implementations;

internal class SerializationService : ISerializationService
{
    private const string SerializerVersion = "Sci.NET v0.2";

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

        tensor.Memory.WriteTo(stream);

        stream.Flush();
        stream.Close();
    }

    public void Save<TNumber>(ITensor<TNumber> tensor, Stream stream)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var rank = tensor.Shape.Rank;
        var shape = tensor.Shape.Dimensions;
        var typeBytes = GetDataType<TNumber>();

        stream.WriteString(SerializerVersion);
        stream.WriteValue(rank);
        stream.WriteValue(Unsafe.SizeOf<TNumber>());
        stream.WriteString(typeBytes);

        for (var i = 0; i < rank; i++)
        {
            stream.WriteValue(shape[i]);
        }

        tensor.Memory.WriteTo(stream);

        stream.Flush();
        stream.Close();
    }

    public void Save<TNumber>(ITensor<TNumber> tensor, string path)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var stream = File.OpenWrite(path);

        Save(tensor, stream);
    }

    public void SaveCompressed<TNumber>(ITensor<TNumber> tensor, string path)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var stream = File.OpenWrite(path);

        SaveCompressed(tensor, stream);
    }

    public void SaveCompressed<TNumber>(ITensor<TNumber> tensor, Stream stream)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var compressedStream = new GZipStream(stream, CompressionMode.Compress, true);

        Save(tensor, compressedStream);
    }

    public ITensor<TNumber> Load<TNumber>(Stream stream)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var version = stream.ReadString();
        var rank = stream.ReadValue<int>();
        var bytesPerElement = stream.ReadValue<int>();
        var type = stream.ReadString();
        var shape = new int[rank];

        for (var i = 0; i < rank; i++)
        {
            shape[i] = stream.ReadValue<int>();
        }

        if (version != SerializerVersion)
        {
            throw new NotSupportedException("The tensor was created with a different version of the serializer.");
        }

        if (bytesPerElement != Unsafe.SizeOf<TNumber>() || type != GetDataType<TNumber>())
        {
            throw new InvalidDataException("The data type of the tensor does not match the data type of the serializer.");
        }

        var tensor = new Tensor<TNumber>(new Shape(shape));
        var handle = tensor.Memory;
        var bufferSize = tensor.Shape.ElementCount <= int.MaxValue ? (int)tensor.Shape.ElementCount : int.MaxValue;
        var buffer = new Span<byte>(new byte[bufferSize]);
        var bytesRead = 0;

        while (stream.CanRead && bytesRead < handle.Length * Unsafe.SizeOf<TNumber>())
        {
            var bytes = stream.Read(buffer);

            handle.BlockCopyFrom(
                buffer,
                0,
                bytesRead,
                bytes);

            bytesRead += bytes;
        }

        return tensor;
    }

    public ITensor<TNumber> Load<TNumber>(string path)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var stream = File.OpenRead(path);

        return Load<TNumber>(stream);
    }

    public ITensor<TNumber> LoadCompressed<TNumber>(string path)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var stream = File.OpenRead(path);

        return LoadCompressed<TNumber>(stream);
    }

    public ITensor<TNumber> LoadCompressed<TNumber>(Stream stream)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var compressedStream = new GZipStream(stream, CompressionMode.Decompress);

        return Load<TNumber>(compressedStream);
    }

    public Dictionary<string, ITensor<TNumber>> LoadSafeTensors<TNumber>(string path)
        where TNumber : unmanaged, INumber<TNumber>
    {
        using var stream = File.OpenRead(path);

        return LoadSafeTensors<TNumber>(stream);
    }

    public Dictionary<string, ITensor<TNumber>> LoadSafeTensors<TNumber>(Stream stream)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var headerLength = stream.ReadValue<long>();
        var headerBuffer = new byte[headerLength];
        _ = stream.Read(headerBuffer);
        var headerString = Encoding.UTF8.GetString(headerBuffer);
        var header = JsonConvert.DeserializeObject<Dictionary<string, SafetensorsHeaderValue>>(headerString)
                     ?? throw new FileLoadException("The header could not be deserialized.");
        var dtypeString = GetSafetensorsDtypeString<TNumber>();
        headerLength += sizeof(long);

        if (header.Any(x => x.Value.Dtype != dtypeString))
        {
            throw new NotSupportedException($"The data type ({header.First().Value.Dtype}) of the tensor does not match the data type of the serializer.");
        }

        var tensors = header.ToDictionary(x => x.Key, x => (ITensor<TNumber>)new Tensor<TNumber>(new Shape(x.Value.Shape.ToArray())));

        try
        {
            foreach (var tensor in header)
            {
                LoadSafetensorData(stream, tensors[tensor.Key], headerLength, tensor.Value);
            }
        }
        catch
        {
            foreach (var tensor in tensors)
            {
                tensor.Value.Dispose();
            }

            throw;
        }

        stream.Close();

        return tensors;
    }

    public void SaveSafeTensors<TNumber>(Dictionary<string, ITensor<TNumber>> tensors, string path)
        where TNumber : unmanaged, INumber<TNumber>
    {
        if (File.Exists(path))
        {
            File.Delete(path);
        }

        using var fileStream = new FileStream(
            path,
            FileMode.CreateNew,
            FileAccess.Write,
            FileShare.None,
            4096,
            FileOptions.RandomAccess);

        SaveSafeTensors(tensors, fileStream);
    }

    public void SaveSafeTensors<TNumber>(Dictionary<string, ITensor<TNumber>> tensors, Stream stream)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var offset = 0L;

        var tensorMetadata = tensors.ToDictionary(
            x => x.Key,
            x =>
            {
                var initialOffset = offset;
                var dataLength = x.Value.Shape.ElementCount * Unsafe.SizeOf<TNumber>();
                offset += dataLength;

                return new SafetensorsHeaderValue
                {
                    Shape = x.Value.Shape.ToList(),
                    Dtype = GetSafetensorsDtypeString<TNumber>(),
                    DataOffsets = new List<long>
                    {
                        initialOffset,
                        initialOffset + dataLength
                    }
                };
            });

        var metadataBuffer = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(tensorMetadata));
        var paddingBytes = (8 - (metadataBuffer.Length % 8)) % 8;

        if (paddingBytes > 0)
        {
            Array.Resize(ref metadataBuffer, metadataBuffer.Length + paddingBytes);
        }

        using var writer = new BinaryWriter(stream);

        writer.Write(BitConverter.GetBytes((long)metadataBuffer.Length));
        writer.Write(metadataBuffer.AsSpan());

        foreach (var tensor in tensors)
        {
            tensor.Value.Memory.WriteTo(writer.BaseStream);
        }

        stream.Flush();
        stream.Close();
    }

    private static void LoadSafetensorData<TNumber>(Stream stream, ITensor<TNumber> tensor, long headerLength, SafetensorsHeaderValue tensorDescriptor)
        where TNumber : unmanaged, INumber<TNumber>
    {
        var storedDataLength = tensorDescriptor.DataOffsets[1] - tensorDescriptor.DataOffsets[0];
        var storedElementSize = storedDataLength / tensor.Shape.ElementCount;
        var handle = tensor.Memory.ToSystemMemory();

        InvalidOperationExceptionHelper.ThrowIfNotEqual(storedElementSize, Unsafe.SizeOf<TNumber>(), "The stored element size does not match the element size of the tensor.");
        InvalidOperationExceptionHelper.ThrowIfNotEqual(storedDataLength, handle.Length * Unsafe.SizeOf<TNumber>(), "The stored data length is not equal to the tensor data length.");

        handle.ReadElementsFrom(stream, tensorDescriptor.DataOffsets[0] + headerLength, storedDataLength);
    }

    [ExcludeFromCodeCoverage]
    private static string GetSafetensorsDtypeString<TNumber>()
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TNumber.Zero switch
        {
            double => "F64",
            float => "F32",
            Half => "F16",
            BFloat16 => "BF16",
            long => "I64",
            ulong => "U64",
            int => "I32",
            uint => "U32",
            short => "I16",
            ushort => "U16",
            sbyte => "I8",
            byte => "U8",
            bool => "BOOL",
            _ => throw new NotSupportedException()
        };
    }

    [ExcludeFromCodeCoverage]
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

    [ExcludeFromCodeCoverage]
    private static string GetDataType<TNumber>()
        where TNumber : unmanaged, INumber<TNumber>
    {
        return TNumber.Zero switch
        {
            byte => "u1",
            sbyte => "i1",
            ushort => "u2",
            short => "i2",
            uint => "u4",
            int => "i4",
            ulong => "u8",
            long => "i8",
            Half => "f2",
            float => "f4",
            double => "f8",
            BFloat16 => "bf2",
            _ => "unknown"
        };
    }
}