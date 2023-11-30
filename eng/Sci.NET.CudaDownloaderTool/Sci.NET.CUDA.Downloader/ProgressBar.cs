﻿namespace Sci.NET.CUDA.Downloader;

using System;
using System.Text;
using System.Threading;

/// <summary>
/// An ASCII progress bar https://gist.github.com/DanielSWolf/0ab6a96899cc5377bf54
/// </summary>
public class ProgressBar : IDisposable, IProgress<ProgressDescriptor>
{
    private const int BlockCount = 10;
    private const string Animation = @"|/-\";

    private readonly TimeSpan _animationInterval = TimeSpan.FromSeconds(1.0 / 16);
    private readonly Timer _timer;

    private double _currentProgress = 0;
    private string _doneText = string.Empty;
    private string _ofText = string.Empty;
    private string _currentText = string.Empty;
    private bool _disposed = false;
    private int _animationIndex = 0;

    public ProgressBar()
    {
        _timer = new Timer(TimerHandler);

        // A progress bar is only for temporary display in a console window.
        // If the console output is redirected to a file, draw nothing.
        // Otherwise, we'll end up with a lot of garbage in the target file.
        if (!Console.IsOutputRedirected)
        {
            ResetTimer();
        }
    }

    public void Report(ProgressDescriptor descriptor)
    {
        // Make sure value is in [0..1] range
        var value = Math.Max(0, Math.Min(1, descriptor.Progress));
        Interlocked.Exchange(ref _currentProgress, value);
        Interlocked.Exchange(ref _doneText, descriptor.Done);
        Interlocked.Exchange(ref _ofText, descriptor.Total);
    }

    private void TimerHandler(object state)
    {
        lock (_timer)
        {
            if (_disposed) return;

            var progressBlockCount = (int) (_currentProgress * BlockCount);
            var percent = (int) (_currentProgress * 100);
            var text = $"[{new string('#', progressBlockCount)}{new string('-', BlockCount - progressBlockCount)}] {percent,3}% {Animation[_animationIndex++ % Animation.Length]} {_doneText}/{_ofText}";
            UpdateText(text);

            ResetTimer();
        }
    }

    private void UpdateText(string text)
    {
        // Get length of common portion
        var commonPrefixLength = 0;
        var commonLength = Math.Min(_currentText.Length, text.Length);

        while (commonPrefixLength < commonLength && text[commonPrefixLength] == _currentText[commonPrefixLength])
        {
            commonPrefixLength++;
        }

        // Backtrack to the first differing character
        var outputBuilder = new StringBuilder();
        outputBuilder.Append('\b', _currentText.Length - commonPrefixLength);

        // Output new suffix
        outputBuilder.Append(text.AsSpan(commonPrefixLength));

        // If the new text is shorter than the old one: delete overlapping characters
        var overlapCount = _currentText.Length - text.Length;

        if (overlapCount > 0)
        {
            outputBuilder.Append(' ', overlapCount);
            outputBuilder.Append('\b', overlapCount);
        }

        Console.Write(outputBuilder);
        _currentText = text;
    }

    private void ResetTimer()
    {
        _timer.Change(_animationInterval, TimeSpan.FromMilliseconds(-1));
    }

    public void Dispose()
    {
        lock (_timer)
        {
            _disposed = true;
            UpdateText(string.Empty);
        }
    }
}