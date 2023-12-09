// See https://aka.ms/new-console-template for more information

using System.IO.Compression;
using System.Security.Cryptography;
using Humanizer;
using Humanizer.Bytes;
using Newtonsoft.Json;
using SharpCompress.Common;
using SharpCompress.Readers;

namespace Sci.NET.CUDA.Downloader;

public class Program
{
    private const string BaseUrl = "https://developer.download.nvidia.com/compute/cuda/redist/";
    private const string IndexName = "redistrib_";
    private const string LinuxX8664 = "linux-x86_64";
    private const string LinuxPpc64Le = "linux-ppc64le";
    private const string LinuxSbsa = "linux-sbsa";
    private const string WindowsX8664 = "windows-x86_64";

    public static async Task Main(string[] args)
    {
        var version = args[0];
        var outputRootPath = args[1];
        var url = $"{BaseUrl}{IndexName}{version}.json";
        var json = await GetVersionSpec(url);

        if (json is null)
        {
            Console.WriteLine($"Could not find index for version. {version}");
            return;
        }

        var tasks = GetDownloadsFromVersionSpec(json);
        var downloadTasks = tasks.Select(x => x.Select(y => y)).SelectMany(x => x).ToList();

        var archivesToExtract = new List<KeyValuePair<string, string>>();
        var i = 0;

        // Download all files
        foreach (var task in downloadTasks)
        {
            var outputDirectory = Path.Combine(Environment.CurrentDirectory, "Downloads", task.Key);

            if (!Directory.Exists(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }

            var outputFileName = Path.Combine(outputDirectory, task.Value.OutputFileName);

            Console.WriteLine($"{i} of {downloadTasks.Count} ");
            Console.Write($"Downloading {task.Value.Name} {task.Key}");

            await DownloadFile($"{BaseUrl}{task.Value.Url}", outputFileName, task.Value.Hash);

            Console.WriteLine("Download complete\n");

            archivesToExtract.Add(new KeyValuePair<string, string>(task.Key, outputFileName));

            i++;
        }

        i = 0;

        // Extract all files
        Parallel.ForEach(
            archivesToExtract,
            new ParallelOptions
            {
                MaxDegreeOfParallelism = 4 // More than 4 threads causes BSOD on windows?
            },
            archive =>
                //foreach (var archive in archivesToExtract)
            {
                var outputDirectory = Path.Combine(Environment.CurrentDirectory, "Extracted", archive.Key);

                if (!Directory.Exists(outputDirectory))
                {
                    Directory.CreateDirectory(outputDirectory);
                }

                Console.Write($"{i} of {archivesToExtract.Count} ");
                Console.WriteLine($"Extracting {archive.Key}/{Path.GetFileName(archive.Value)}");

                ExtractArchive(archive.Value, outputDirectory);

                Console.WriteLine("Extraction complete\n");

                i++;
            });

        // Move files to final destination
        foreach (var archive in Directory.EnumerateDirectories(Path.Combine(Environment.CurrentDirectory, "Extracted", LinuxX8664)))
        {
            MoveFolder(archive, Path.Combine(outputRootPath, LinuxX8664));
        }

        foreach (var archive in Directory.EnumerateDirectories(Path.Combine(Environment.CurrentDirectory, "Extracted", LinuxPpc64Le)))
        {
            MoveFolder(archive, Path.Combine(outputRootPath, LinuxPpc64Le));
        }

        foreach (var archive in Directory.EnumerateDirectories(Path.Combine(Environment.CurrentDirectory, "Extracted", LinuxSbsa)))
        {
            MoveFolder(archive, Path.Combine(outputRootPath, LinuxSbsa));
        }

        foreach (var archive in Directory.EnumerateDirectories(Path.Combine(Environment.CurrentDirectory, "Extracted", WindowsX8664)))
        {
            MoveFolder(archive, Path.Combine(outputRootPath, WindowsX8664));
        }
    }

    private static void MoveFolder(string source, string destination)
    {
        if (Directory.Exists(source))
        {
            foreach (var dirPath in Directory.GetDirectories(source, "*", SearchOption.AllDirectories))
            {
                var destDirPath = dirPath.Replace(source, destination);
                Directory.CreateDirectory(destDirPath);
            }

            foreach (var filePath in Directory.GetFiles(source, "*.*", SearchOption.AllDirectories))
            {
                var destFilePath = filePath.Replace(source, destination);

                if (File.Exists(destFilePath))
                {
                    File.Delete(destFilePath);
                }

                File.Move(filePath, destFilePath);
            }

            Directory.Delete(source, true);
        }
        else
        {
            Console.WriteLine($"Subfolder {source} does not exist under {destination}");
        }
    }

    private static void ExtractArchive(string target, string destination)
    {
        using var reader = File.Open(target, FileMode.Open);

        if (target.EndsWith(".zip"))
        {
            using var archive = new ZipArchive(reader);
            archive.ExtractToDirectory(destination, true);
        }
        else if (target.EndsWith(".tar.xz"))
        {
            using var archive = ReaderFactory.Open(reader);

            while (archive.MoveToNextEntry())
            {
                if (!archive.Entry.IsDirectory)
                {
                    archive.WriteEntryToDirectory(
                        destination,
                        new ExtractionOptions
                        {
                            ExtractFullPath = true,
                            Overwrite = true,
                            WriteSymbolicLink = (_, _) => { }
                        });
                }
            }
        }
    }

    private static async Task<VersionSpec?> GetVersionSpec(string url)
    {
        using var httpClient = new HttpClient();
        var response = await httpClient.GetAsync(url);
        var content = await response.Content.ReadAsStringAsync();
        var json = JsonConvert.DeserializeObject<VersionSpec>(content);
        return json;
    }

    private static async Task DownloadFile(string remoteFilename, string localFilename, string hash)
    {
        if (File.Exists(localFilename))
        {
            Console.WriteLine("File already exists, checking hash");

            if (ComputeHash(localFilename) != hash)
            {
                Console.WriteLine("Hash mismatch");
                File.Delete(localFilename);
            }
            else
            {
                return;
            }
        }

        using var httpClientDownload = new HttpClientDownloadWithProgress(
            remoteFilename,
            localFilename);
        using var progress = new ProgressBar();

        httpClientDownload.ProgressChanged += (total, done, progressPercentage) =>
        {
            var byteSize = new ByteSize(total ?? 0).Humanize();
            var doneSize = new ByteSize(done).Humanize();
            progress.Report(
                new ProgressDescriptor
                {
                    Progress = progressPercentage / 100 ?? 0,
                    Done = doneSize,
                    Total = byteSize
                });
        };

        await httpClientDownload.StartDownload();

        Console.WriteLine("Download complete, checking hash");

        if (ComputeHash(localFilename) != hash)
        {
            Console.WriteLine("WARNING: Hash mismatch");
        }
    }

    private static string ComputeHash(string hash)
    {
        using var sha256 = SHA256.Create();
        using var stream = File.OpenRead(hash);
        var hashBytes = sha256.ComputeHash(stream);
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
    }

    public static List<List<KeyValuePair<string, DownloadTask>>> GetDownloadsFromVersionSpec(VersionSpec versionSpec)
    {
        var tasks = new List<List<KeyValuePair<string, DownloadTask>>>
        {
            GetTask(versionSpec.CudaCccl),
            GetTask(versionSpec.CudaCudart),
            GetTask(versionSpec.CudaCuObjDump),
            GetTask(versionSpec.CudaCupti),
            GetTask(versionSpec.CudaCuxxfilt),
            //GetTask(versionSpec.CudaDemoSuite),
            //GetTask(versionSpec.CudaDocumentation),
            GetTask(versionSpec.CudaGdb),
            //GetTask(versionSpec.CudaNsight),
            GetTask(versionSpec.CudaNvcc),
            GetTask(versionSpec.CudaNvdisasm),
            GetTask(versionSpec.CudaNvmlDev),
            GetTask(versionSpec.CudaNvprof),
            GetTask(versionSpec.CudaNvprune),
            GetTask(versionSpec.CudaNvrtc),
            GetTask(versionSpec.CudaNvtx),
            GetTask(versionSpec.CudaNvvp),
            GetTask(versionSpec.CudaOpencl),
            GetTask(versionSpec.CudaProfilerApi),
            GetTask(versionSpec.CudaSanitizerApi),
            GetTask(versionSpec.Fabricmanager),
            GetTask(versionSpec.CuBlas),
            GetTask(versionSpec.CuFft),
            GetTask(versionSpec.CuFile),
            GetTask(versionSpec.CuRand),
            GetTask(versionSpec.CuSolver),
            GetTask(versionSpec.CuSparse),
            GetTask(versionSpec.Npp),
            GetTask(versionSpec.NvidiaNscq),
            GetTask(versionSpec.NvJitLink),
            GetTask(versionSpec.NvJpeg),
            //GetTask(versionSpec.NsightCompute),
            //GetTask(versionSpec.NsightSystems),
            //GetTask(versionSpec.NsightVse),
            //GetTask(versionSpec.NvidiaDriver),
            GetTask(versionSpec.NvidiaFs),
            //GetTask(versionSpec.VisualStudioIntegration),
        };

        return tasks;
    }

    public static List<KeyValuePair<string, DownloadTask>> GetTask(ApiComponent component)
    {
        var dict = new List<KeyValuePair<string, DownloadTask>>();

        if (component.LinuxX8664 is not null)
        {
            dict.Add(
                new KeyValuePair<string, DownloadTask>(
                    LinuxX8664,
                    new DownloadTask
                    {
                        Url = component.LinuxX8664.RelativePath,
                        Name = component.Name,
                        OutputFileName = Path.GetFileName(component.LinuxX8664.RelativePath),
                        Hash = component.LinuxX8664.Sha256
                    }));
        }

        if (component.WindowsX8664 is not null)
        {
            dict.Add(
                new KeyValuePair<string, DownloadTask>(
                    WindowsX8664,
                    new DownloadTask
                    {
                        Url = component.WindowsX8664.RelativePath,
                        Name = component.Name,
                        OutputFileName = Path.GetFileName(component.WindowsX8664.RelativePath),
                        Hash = component.WindowsX8664.Sha256
                    }));
        }

        dict.Add(
            new KeyValuePair<string, DownloadTask>(
                "Licenses",
                new DownloadTask
                {
                    Url = component.LicensePath,
                    Name = component.Name,
                    OutputFileName = $"{component.Name}-{Path.GetFileName(component.LicensePath)}"
                }));

        return dict;
    }
}