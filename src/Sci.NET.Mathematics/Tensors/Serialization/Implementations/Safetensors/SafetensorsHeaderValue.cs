// Copyright (c) Sci.NET Foundation. All rights reserved.
// Licensed under the Apache 2.0 license. See LICENSE file in the project root for full license information.

using Newtonsoft.Json;

namespace Sci.NET.Mathematics.Tensors.Serialization.Implementations.Safetensors;

internal class SafetensorsHeaderValue
{
    [JsonProperty("dtype")]
    public required string Dtype { get; set; }

    [JsonProperty("shape")]
    public required List<int> Shape { get; set; }

    [JsonProperty("data_offsets")]
    public required List<long> DataOffsets { get; set; }
}