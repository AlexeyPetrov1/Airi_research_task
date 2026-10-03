param([Parameter(Mandatory=$true)][string]$OutputDir)
$receiptPath = Join-Path $OutputDir 'process_exit.json'
$metricsPath = Join-Path $OutputDir 'host_resource_usage.json'
$peakUsedBytes = [int64]0
$startedUtc = [DateTime]::UtcNow.ToString('o')
while ($true) {
    $taskOs = Get-CimInstance Win32_OperatingSystem
    $totalBytes = [int64]$taskOs.TotalVisibleMemorySize * 1024
    $freeBytes = [int64]$taskOs.FreePhysicalMemory * 1024
    $peakUsedBytes = [Math]::Max($peakUsedBytes, $totalBytes - $freeBytes)
    $result = [ordered]@{
        monitor_started_utc = $startedUtc
        sample_period_seconds = 2
        host_physical_ram_bytes = $totalBytes
        peak_host_physical_used_bytes = $peakUsedBytes
        current_host_free_bytes = $freeBytes
        note = 'Whole Windows host, including other processes. Monitor started during first model load; WSL process monitoring covers the full generation.'
    }
    $result | ConvertTo-Json | Set-Content -LiteralPath $metricsPath -Encoding UTF8
    if (Test-Path -LiteralPath $receiptPath) {
        try {
            $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
            if (-not $receipt.running) { break }
        } catch { }
    }
    Start-Sleep -Seconds 2
}
