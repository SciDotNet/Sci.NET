FROM mcr.microsoft.com/dotnet/runtime:8.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY ["Sci.NET.CUDA.Downloader/Sci.NET.CUDA.Downloader.csproj", "Sci.NET.CUDA.Downloader/"]
RUN dotnet restore "Sci.NET.CUDA.Downloader/Sci.NET.CUDA.Downloader.csproj"
COPY . .
WORKDIR "/src/Sci.NET.CUDA.Downloader"
RUN dotnet build "Sci.NET.CUDA.Downloader.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "Sci.NET.CUDA.Downloader.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "Sci.NET.CUDA.Downloader.dll"]
