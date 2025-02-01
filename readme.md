![Logo](https://github.com/SciDotNet/Sci.NET/blob/main/eng/build-props/images/icon-128.png)
[![.NET](https://github.com/SciDotNet/Sci.NET/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SciDotNet/Sci.NET/actions/workflows/ci.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=SciDotNet_Sci.NET&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=SciDotNet_Sci.NET)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=SciDotNet_Sci.NET&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=SciDotNet_Sci.NET)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=SciDotNet_Sci.NET&metric=coverage)](https://sonarcloud.io/summary/new_code?id=SciDotNet_Sci.NET)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=SciDotNet_Sci.NET&metric=bugs)](https://sonarcloud.io/summary/new_code?id=SciDotNet_Sci.NET)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=SciDotNet_Sci.NET&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=SciDotNet_Sci.NET)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=SciDotNet_Sci.NET&metric=vulnerabilities)](https://sonarcloud.io/summary/new_code?id=SciDotNet_Sci.NET)
[![Technical Debt](https://sonarcloud.io/api/project_badges/measure?project=SciDotNet_Sci.NET&metric=sqale_index)](https://sonarcloud.io/summary/new_code?id=SciDotNet_Sci.NET)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=SciDotNet_Sci.NET&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=SciDotNet_Sci.NET)

# Sci.NET (Early Development)

Sci.NET is a scientific computing library for .NET with a familiar API similar to NumPy and PyTorch. Currently, the
library is in early development and is not ready for production use.

Currently, only the managed CPU backend is working which is used as a proof of concept and regression testing for other
backends.

### Currently Supported Features:
- Arithmetic operations
- Contractions
- Reductions
- Trigonometric functions
- Automatic differentiation (Reverse Mode, WIP)

## License

[Apache License 2.0](https://github.com/SciDotNet/Sci.NET/blob/main/LICENSE/)

**Some packages include third-party components with specific licensing requirements**

## Requirements

### Base Requirements

- .NET 8 SDK
- That's it!

## Packages

There are a number of packages offered with Sci.NET. The base package is the `Sci.NET.Mathematics` package.

Work has been started on the CUDA backend, but is not yet ready. There will be redistributable packages for CUDA for
Windows and Linux.

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

[Documentation](http://docs.scidotnet.org/) (Not yet available, depends on the build pipeline)

## Roadmap
- Vectorized operations for CPU backend.
- Re-implement the machine learning API for `Sci.NET.MachineLearning`.
- Better hardware acceleration using intrinsics.
- CUDA, Vulkan and MKL backend (quite far in the future).
- More comprehensive documentation.
- Better support for datasets and data loading.

## Third Party

- [NVIDIA CUDA](https://docs.nvidia.com/cuda/eula/index.html)
- [ImageSharp](https://github.com/SixLabors/ImageSharp)