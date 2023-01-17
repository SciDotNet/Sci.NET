# Sci.NET

##### WARNING - Sci.NET is currently in a very early stage of development. The API is not stable and will change frequently. Use at your own risk.

## About 
Sci.NET is a scientific computing library designed to be used with .NET with a focus on performance and ease of use. The project is
heavily reliant on tensors and tensor operations, and much time has been spent optimizing the performance of tensor operations, while
also providing a simple and intuitive API.

## Features
- [x] Tensor contraction
- [ ] Point-wise tensor operations
- [ ] Native, multi-threaded BLAS engine
- [ ] CUDA BLAS engine
- [ ] Deep learning API (similar to PyToch or TensorFlow)
- [ ] Automatic differentiation
- [ ] Data visualization tools
- [ ] Distributed computing tools

Some of these features are currently in development but are not available in the main repository yet, as architecture, technologies 
and libraries are still being decided upon.

## Requirements
- .NET 6 Developer Pack
- CMake (3.26 or higher)
- CUDA Toolkit (12.0 or higher, optional)
- MSBuild command line tools
- C++ compiler (MSVC)

## Getting Started

First you will need to clone the repo, build the CMake project and generate the solution files.
```
git https://github.com/sciDotNet/Sci.NET --recursive
cd Sci.NET
./build.cmd
```
You should now have a solution file with the name `Sci.NET.sln` in the root directory. Open this file in Visual Studio and build the project.

## Contributing

Please note that this project uses [SonarCloud](https://sonarcloud.io/dashboard?id=sciDotNet_Sci.NET) for code quality analysis as well as 
roslyn analyzers to enforce code style. Please make sure that your code passes these checks and all unit tests pass before submitting a pull request.