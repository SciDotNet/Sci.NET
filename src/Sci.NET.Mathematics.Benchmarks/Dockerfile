FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["src/Sci.NET.Mathematics.Benchmarks/Sci.NET.Mathematics.Benchmarks.csproj", "src/Sci.NET.Mathematics.Benchmarks/"]
RUN dotnet restore "src/Sci.NET.Mathematics.Benchmarks/Sci.NET.Mathematics.Benchmarks.csproj"
COPY . .
WORKDIR "/src/src/Sci.NET.Mathematics.Benchmarks"
RUN dotnet build "Sci.NET.Mathematics.Benchmarks.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "Sci.NET.Mathematics.Benchmarks.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "Sci.NET.Mathematics.Benchmarks.dll"]
