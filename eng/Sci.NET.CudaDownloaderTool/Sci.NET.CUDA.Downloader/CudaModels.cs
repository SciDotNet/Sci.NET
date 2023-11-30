using Newtonsoft.Json;

namespace Sci.NET.CUDA.Downloader;

public class ApiComponent
{
    [JsonProperty("name")]
    public string Name { get; set; }

    [JsonProperty("license")]
    public string License { get; set; }

    [JsonProperty("license_path")]
    public string LicensePath { get; set; }

    [JsonProperty("version")]
    public string Version { get; set; }

    [JsonProperty("linux-x86_64")]
    public ItemInfo? LinuxX8664 { get; set; }

    [JsonProperty("linux-ppc64le")]
    public ItemInfo? LinuxPpc64le { get; set; }

    [JsonProperty("linux-sbsa")]
    public ItemInfo? LinuxSbsa { get; set; }

    [JsonProperty("windows-x86_64")]
    public ItemInfo? WindowsX8664 { get; set; }
}

public class VersionSpec
{
    [JsonProperty("release_date")]
    public string ReleaseDate { get; set; }

    [JsonProperty("release_label")]
    public string ReleaseLabel { get; set; }

    [JsonProperty("release_product")]
    public string ReleaseProduct { get; set; }

    [JsonProperty("cuda_cccl")]
    public ApiComponent CudaCccl { get; set; }

    [JsonProperty("cuda_cudart")]
    public ApiComponent CudaCudart { get; set; }

    [JsonProperty("cuda_cuobjdump")]
    public ApiComponent CudaCuObjDump { get; set; }

    [JsonProperty("cuda_cupti")]
    public ApiComponent CudaCupti { get; set; }

    [JsonProperty("cuda_cuxxfilt")]
    public ApiComponent CudaCuxxfilt { get; set; }

    [JsonProperty("cuda_demo_suite")]
    public ApiComponent CudaDemoSuite { get; set; }

    [JsonProperty("cuda_documentation")]
    public ApiComponent CudaDocumentation { get; set; }

    [JsonProperty("cuda_gdb")]
    public ApiComponent CudaGdb { get; set; }

    [JsonProperty("cuda_nsight")]
    public ApiComponent CudaNsight { get; set; }

    [JsonProperty("cuda_nvcc")]
    public ApiComponent CudaNvcc { get; set; }

    [JsonProperty("cuda_nvdisasm")]
    public ApiComponent CudaNvdisasm { get; set; }

    [JsonProperty("cuda_nvml_dev")]
    public ApiComponent CudaNvmlDev { get; set; }

    [JsonProperty("cuda_nvprof")]
    public ApiComponent CudaNvprof { get; set; }

    [JsonProperty("cuda_nvprune")]
    public ApiComponent CudaNvprune { get; set; }

    [JsonProperty("cuda_nvrtc")]
    public ApiComponent CudaNvrtc { get; set; }

    [JsonProperty("cuda_nvtx")]
    public ApiComponent CudaNvtx { get; set; }

    [JsonProperty("cuda_nvvp")]
    public ApiComponent CudaNvvp { get; set; }

    [JsonProperty("cuda_opencl")]
    public ApiComponent CudaOpencl { get; set; }

    [JsonProperty("cuda_profiler_api")]
    public ApiComponent CudaProfilerApi { get; set; }

    [JsonProperty("cuda_sanitizer_api")]
    public ApiComponent CudaSanitizerApi { get; set; }

    [JsonProperty("fabricmanager")]
    public ApiComponent Fabricmanager { get; set; }

    [JsonProperty("libcublas")]
    public ApiComponent CuBlas { get; set; }

    [JsonProperty("libcufft")]
    public ApiComponent CuFft { get; set; }

    [JsonProperty("libcufile")]
    public ApiComponent CuFile { get; set; }

    [JsonProperty("libcurand")]
    public ApiComponent CuRand { get; set; }

    [JsonProperty("libcusolver")]
    public ApiComponent CuSolver { get; set; }

    [JsonProperty("libcusparse")]
    public ApiComponent CuSparse { get; set; }

    [JsonProperty("libnpp")]
    public ApiComponent Npp { get; set; }

    [JsonProperty("libnvidia_nscq")]
    public ApiComponent NvidiaNscq { get; set; }

    [JsonProperty("libnvjitlink")]
    public ApiComponent NvJitLink { get; set; }

    [JsonProperty("libnvjpeg")]
    public ApiComponent NvJpeg { get; set; }

    [JsonProperty("nsight_compute")]
    public ApiComponent NsightCompute { get; set; }

    [JsonProperty("nsight_systems")]
    public ApiComponent NsightSystems { get; set; }

    [JsonProperty("nsight_vse")]
    public ApiComponent NsightVse { get; set; }

    [JsonProperty("nvidia_driver")]
    public ApiComponent NvidiaDriver { get; set; }

    [JsonProperty("nvidia_fs")]
    public ApiComponent NvidiaFs { get; set; }

    [JsonProperty("visual_studio_integration")]
    public ApiComponent VisualStudioIntegration { get; set; }
}

public class ItemInfo
{
    [JsonProperty("relative_path")]
    public string RelativePath { get; set; }

    [JsonProperty("sha256")]
    public string Sha256 { get; set; }

    [JsonProperty("md5")]
    public string Md5 { get; set; }

    [JsonProperty("size")]
    public string Size { get; set; }
}

