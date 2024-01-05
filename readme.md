
![Logo](https://github.com/SciDotNet/Sci.NET/blob/main/eng/build-props/images/icon-128.png)
[![.NET](https://github.com/SciDotNet/Sci.NET/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SciDotNet/Sci.NET/actions/workflows/ci.yml)

# Sci.NET

Sci.NET is a scientific computing library for .NET with a familiar API similar to NumPy and PyTorch. Currently, the library is in early development and is not ready for production use. 

Currently, only the managed CPU backend is working which is used as a proof of concept and regression testing for other backends.

The subset of working features include:
* Most Tensor Operations (Arithmetic, Trigonometry, etc.)
* Most Tensor Transformations (Reshape, Transpose, Permute, etc.)
* Most Contraction aliases (Contract, Dot Product, Matrix Multiply, Inner Product)

Known to be broken:
* Most of the Machine Learning API

The library is designed to be extensible, easy to use and hardware agnostic with a focus on performance and heterogeneous computing.

## License

[Apache License 2.0](https://github.com/SciDotNet/Sci.NET/blob/main/LICENSE/)

**Some packages include third-party components with specific licensing requirements**

## Requirements
### Base Requirements
- .NET 8 SDK
### Development Requirements
- CUDA 12.3


## Packages

There are a number of packages offered with Sci.NET. The base package is the `Sci.NET.Mathematics` package.

Work has been started on the CUDA backend, but is not yet ready. There will be redistributable packages for CUDA for Windows and Linux. 

The `Sci.NET.CUDA.Redist-*platform*` packages include the CUDA Runtime. *See third-party licences*.
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
- Working build pipeline.
- Full testing of `Sci.NET.Mathematics`.
- More comprehensive machine learning API for `Sci.NET.MachineLearning`.
- Better hardware acceleration using intrinsics.
- CUDA, Vulkan and MKL backend.
- More comprehensive documentation.
- Better support for datasets.

Far in the future
- Native image processing library.


## Third Party

- [NVIDIA CUDA](https://docs.nvidia.com/cuda/eula/index.html)
- [ImageSharp](https://github.com/SixLabors/ImageSharp)