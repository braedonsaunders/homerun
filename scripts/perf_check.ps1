# perf_check.ps1 - System Performance Check

Write-Host "============================================"
Write-Host "  SYSTEM PERFORMANCE CHECK"
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================"
Write-Host ""

# 1. CPU Usage (overall percentage)
Write-Host "--- CPU USAGE ---"
try {
    $cpu = Get-CimInstance -ClassName Win32_Processor | Measure-Object -Property LoadPercentage -Average
    Write-Host "Overall CPU Usage: $($cpu.Average)%"
} catch {
    Write-Host "Could not retrieve CPU usage: $_"
}
Write-Host ""

# 2. Memory Usage (total, used, free, percentage)
Write-Host "--- MEMORY USAGE ---"
try {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $totalMB = [math]::Round($os.TotalVisibleMemorySize / 1024, 2)
    $freeMB = [math]::Round($os.FreePhysicalMemory / 1024, 2)
    $usedMB = [math]::Round($totalMB - $freeMB, 2)
    $pct = [math]::Round(($usedMB / $totalMB) * 100, 1)
    Write-Host "Total:      $totalMB MB ($([math]::Round($totalMB / 1024, 2)) GB)"
    Write-Host "Used:       $usedMB MB ($([math]::Round($usedMB / 1024, 2)) GB)"
    Write-Host "Free:       $freeMB MB ($([math]::Round($freeMB / 1024, 2)) GB)"
    Write-Host "Usage:      $pct%"
} catch {
    Write-Host "Could not retrieve memory info: $_"
}
Write-Host ""

# 3. Python processes
Write-Host "--- PYTHON PROCESSES ---"
try {
    $pyProcs = Get-Process -Name python -ErrorAction SilentlyContinue
    if ($pyProcs) {
        $pyCount = ($pyProcs | Measure-Object).Count
        $pyMemMB = [math]::Round(($pyProcs | Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB, 2)
        Write-Host "Count:      $pyCount"
        Write-Host "Total Mem:  $pyMemMB MB"
    } else {
        Write-Host "No python.exe processes running."
    }
} catch {
    Write-Host "Error checking python processes: $_"
}
Write-Host ""

# 4. Postgres processes
Write-Host "--- POSTGRES PROCESSES ---"
try {
    $pgProcs = Get-Process -Name postgres -ErrorAction SilentlyContinue
    if ($pgProcs) {
        $pgCount = ($pgProcs | Measure-Object).Count
        $pgMemMB = [math]::Round(($pgProcs | Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB, 2)
        Write-Host "Count:      $pgCount"
        Write-Host "Total Mem:  $pgMemMB MB"
    } else {
        Write-Host "No postgres.exe processes running."
    }
} catch {
    Write-Host "Error checking postgres processes: $_"
}
Write-Host ""

# 5. Node processes
Write-Host "--- NODE PROCESSES ---"
try {
    $nodeProcs = Get-Process -Name node -ErrorAction SilentlyContinue
    if ($nodeProcs) {
        $nodeCount = ($nodeProcs | Measure-Object).Count
        $nodeMemMB = [math]::Round(($nodeProcs | Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB, 2)
        Write-Host "Count:      $nodeCount"
        Write-Host "Total Mem:  $nodeMemMB MB"
    } else {
        Write-Host "No node.exe processes running."
    }
} catch {
    Write-Host "Error checking node processes: $_"
}
Write-Host ""

# 6. Top 10 processes by memory usage
Write-Host "--- TOP 10 PROCESSES BY MEMORY ---"
try {
    $top = Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 10
    Write-Host ("{0,-8} {1,-30} {2,12}" -f "PID", "Name", "Mem (MB)")
    Write-Host ("{0,-8} {1,-30} {2,12}" -f "---", "----", "--------")
    foreach ($p in $top) {
        $memMB = [math]::Round($p.WorkingSet64 / 1MB, 2)
        Write-Host ("{0,-8} {1,-30} {2,12}" -f $p.Id, $p.ProcessName, $memMB)
    }
} catch {
    Write-Host "Error listing top processes: $_"
}
Write-Host ""

# 7. WT_SESSION environment variable
Write-Host "--- WINDOWS TERMINAL DETECTION ---"
$wtSession = $env:WT_SESSION
if ($wtSession) {
    Write-Host "WT_SESSION is SET: $wtSession"
    Write-Host "Running inside Windows Terminal: YES"
} else {
    Write-Host "WT_SESSION is NOT set."
    Write-Host "Running inside Windows Terminal: NO (or not detectable from this context)"
}
Write-Host ""

# 8. Console buffer size vs window size
Write-Host "--- CONSOLE BUFFER vs WINDOW SIZE ---"
try {
    $bufferWidth = $Host.UI.RawUI.BufferSize.Width
    $bufferHeight = $Host.UI.RawUI.BufferSize.Height
    $windowWidth = $Host.UI.RawUI.WindowSize.Width
    $windowHeight = $Host.UI.RawUI.WindowSize.Height
    Write-Host "Buffer Size:  ${bufferWidth} x ${bufferHeight} (Width x Height)"
    Write-Host "Window Size:  ${windowWidth} x ${windowHeight} (Width x Height)"
    if ($bufferWidth -gt $windowWidth -or $bufferHeight -gt $windowHeight) {
        Write-Host "Note: Buffer is larger than window (scrollback enabled)."
    } else {
        Write-Host "Note: Buffer matches window size."
    }
} catch {
    Write-Host "Could not retrieve console size info: $_"
}
Write-Host ""
Write-Host "============================================"
Write-Host "  CHECK COMPLETE"
Write-Host "============================================"
