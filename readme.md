
![Logo](https://github.com/SciDotNet/Sci.NET/blob/main/eng/build-props/images/icon-small.png)


# Sci.NET

Sci.NET is a scientific computing library for .NET with a familiar API similar to NumPy and PyTorch.


## License

[Apache License 2.0](https://github.com/SciDotNet/Sci.NET/blob/main/LICENSE/)

**Some packages include third-party components with specific licensing requirements**

## Requirements

- .NET 8 SDK
- C++ Compiler (built with MSVC)
- CUDA 12.1
- CUDNN
- OpenMP
- CMake


## Packages

There are a number of packages offered with Sci.NET. The base package is the `Sci.NET.Mathematics` package.

To use CUDA, you must also install the `Sci.NET.Mathematics.CUDA` package and the `Sci.NET.CUDA.Redist-*platform*` package for your architecture and operating system.

The `Sci.NET.CUDA.Redist-*platform*` packages include the CUDA Runtime and CuDNN. *See third-party licences*.
## Build Locally

Setup your environment with the build scripts:

```bash
  git clone https://github.com/SciDotNet/Sci.NET.git
  cd Sci.NET
  ./build-dev.cmd
```

Or to build a release version:

```bash
  git clone https://github.com/SciDotNet/Sci.NET.git
  cd Sci.NET
  ./build-release.cmd
```
## Documentation

[Documentation](https://linktodocumentation)


## Roadmap

- More comprehensive mathematics API for `Sci.NET.Mathematics`.
- More comprehensive machine learning API for `Sci.NET.MachineLearning`.
- Better hardware acceleration using intrinsics.
- CUDA, Vulkan and MKL backend.
- ONNX.
- SafeTensors.
- More comprehensive documentation.


## Third Party

What optimizations did you make in your code? E.g. refactors, performance improvements, accessibility

- [NVIDIA CUDA](https://docs.nvidia.com/cuda/eula/index.html)
- [NVIDIA CuDNN](https://docs.nvidia.com/deeplearning/cudnn/sla/index.html)
- [ImageSharp](https://github.com/SixLabors/ImageSharp)