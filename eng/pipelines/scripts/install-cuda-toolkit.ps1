$version = "12.0"

$downloadFile = "https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe"
$outputFile = "./cuda_toolkit.exe"
$modulesToInstall = "cudart_$version",
                    "cuobjdump_$version",
                    "cuxxfilt_$version",
                    "memcheck_$version",
                    "nvcc_$version",
                    "nvdisasm_$version",
                    "nvml_dev_$version",
                    "nvprune_$version",
                    "nvrtc_$version",
                    "nvtx_$version",
                    "sanitizer_$version",
                    "cublas_$version",
                    "cufft_$version",
                    "curand_$version",
                    "cusolver_$version",
                    "cusparse_$version",
                    "npp_$version",
                    "nvjpeg_$version",
                    "nsight_compute_$version",
                    "nsight_nvtx_$version",
                    "nsight_systems_$version",
                    "nsight_vse_$version",
                    "visual_studio_integration_$version"

Write-Host "Set Up CUDA Toolkit $version"
Write-Host "Downloading $downloadFile to $outputFile"

Invoke-WebRequest $downloadFile -OutFile $outputFile

Write-Host "Installing CUDA Toolkit $version"

$count = 0
$length = $modulesToInstall.Length

foreach ($module in $modulesToInstall) {
    $count = $count + 1
    Write-Host "Installing $module ($count of $length)"
    &$outputFile -s $modulesToInstall
}