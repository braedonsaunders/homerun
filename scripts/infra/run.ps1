# Homerun - Windows Run Script (TUI)
# Run: .\scripts\infra\run.ps1

$ErrorActionPreference = "Stop"

# Navigate to project root (grandparent of scripts\infra\)
Set-Location (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)))

$runServiceSmokeTest = $false
$tuiArgs = @()
foreach ($arg in $args) {
    if ($arg -eq "--services-smoke-test") {
        $runServiceSmokeTest = $true
    } else {
        $tuiArgs += $arg
    }
}

$redisHost = if ($env:REDIS_HOST) { $env:REDIS_HOST } else { "127.0.0.1" }
$redisPort = if ($env:REDIS_PORT) { [int]$env:REDIS_PORT } else { 6379 }
$redisContainerName = if ($env:REDIS_CONTAINER_NAME) { $env:REDIS_CONTAINER_NAME } else { "homerun-redis" }
$redisImage = if ($env:REDIS_IMAGE) { $env:REDIS_IMAGE } else { "redis:7-alpine" }

$postgresHost = if ($env:POSTGRES_HOST) { $env:POSTGRES_HOST } else { "127.0.0.1" }
$postgresPort = if ($env:POSTGRES_PORT) { [int]$env:POSTGRES_PORT } else { 5432 }
$postgresDb = if ($env:POSTGRES_DB) { $env:POSTGRES_DB } else { "homerun" }
$postgresUser = if ($env:POSTGRES_USER) { $env:POSTGRES_USER } else { "homerun" }
$postgresPassword = if ($env:POSTGRES_PASSWORD) { $env:POSTGRES_PASSWORD } else { "homerun" }
$postgresContainerName = if ($env:POSTGRES_CONTAINER_NAME) { $env:POSTGRES_CONTAINER_NAME } else { "homerun-postgres" }
$postgresImage = if ($env:POSTGRES_IMAGE) { $env:POSTGRES_IMAGE } else { "postgres:16-alpine" }
$postgresDataDir = if ($env:POSTGRES_DATA_DIR) { $env:POSTGRES_DATA_DIR } else { Join-Path (Get-Location).Path "data\postgres" }

$script:redisStartedByScript = $false
$script:redisStartMode = ""
$script:redisDockerCreatedByScript = $false
$script:postgresStartedByScript = $false
$script:postgresStartMode = ""
$script:postgresDockerCreatedByScript = $false
$script:databaseUrlWasProvided = [bool]$env:DATABASE_URL

function Test-TcpPort {
    param(
        [string]$TargetHost,
        [int]$Port
    )

    $client = $null
    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $client.ReceiveTimeout = 500
        $client.SendTimeout = 500
        $client.Connect($TargetHost, $Port)
        return $true
    } catch {
        return $false
    } finally {
        if ($client) { $client.Dispose() }
    }
}

function Test-NpcapLoopbackInterference {
    <#
    .SYNOPSIS
    Detect and disable the Npcap Loopback Adapter if present.

    The Npcap Loopback Adapter (installed by Wireshark/Nmap) can interfere with
    TCP connections on 127.0.0.1, causing postgres and other services to appear
    to listen but drop all incoming SYN-ACK packets.  This function checks if
    the adapter is present and enabled, and attempts to disable it (requires
    elevation).  If elevation fails, it warns the user.
    #>
    $adapter = Get-NetAdapter -Name "Npcap Loopback Adapter" -ErrorAction SilentlyContinue
    if (-not $adapter -or $adapter.Status -ne "Up") {
        return  # Not present or already disabled
    }

    Write-Host "Npcap Loopback Adapter detected (enabled). This can block local database connections." -ForegroundColor Yellow
    Write-Host "Attempting to disable it..." -ForegroundColor Yellow

    try {
        # Try directly (works if running elevated)
        Disable-NetAdapter -Name "Npcap Loopback Adapter" -Confirm:$false -ErrorAction Stop
        Write-Host "Npcap Loopback Adapter disabled." -ForegroundColor Green
        return
    } catch {}

    # Try self-elevation
    try {
        $tmpScript = Join-Path $env:TEMP "homerun_disable_npcap.ps1"
        "Disable-NetAdapter -Name 'Npcap Loopback Adapter' -Confirm:`$false -ErrorAction Stop" | Set-Content $tmpScript -Encoding UTF8
        $proc = Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$tmpScript`"" -Wait -PassThru
        Remove-Item $tmpScript -Force -ErrorAction SilentlyContinue
        if ($proc.ExitCode -eq 0) {
            Write-Host "Npcap Loopback Adapter disabled." -ForegroundColor Green
            return
        }
    } catch {}

    Write-Host "Could not disable the Npcap Loopback Adapter automatically." -ForegroundColor Red
    Write-Host "Please disable it manually: Network Settings > Change adapter options > right-click 'Npcap Loopback Adapter' > Disable" -ForegroundColor Yellow
    Write-Host "Or run in an elevated terminal: Disable-NetAdapter -Name 'Npcap Loopback Adapter' -Confirm:`$false" -ForegroundColor Yellow
}

function Wait-ForService {
    param(
        [string]$TargetHost,
        [int]$Port
    )

    for ($i = 0; $i -lt 40; $i++) {
        if (Test-TcpPort -TargetHost $TargetHost -Port $Port) {
            return $true
        }
        Start-Sleep -Milliseconds 250
    }
    return $false
}

function Find-RedisServer {
    $cmd = Get-Command redis-server -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $wellKnown = "C:\Program Files\Redis\redis-server.exe"
    if (Test-Path $wellKnown) { return $wellKnown }

    return $null
}

function Find-PostgresBinDir {
    if ($env:POSTGRES_BIN_DIR) {
        $envBin = $env:POSTGRES_BIN_DIR
        if (
            (Test-Path (Join-Path $envBin "initdb.exe")) -and
            (Test-Path (Join-Path $envBin "pg_ctl.exe"))
        ) {
            return $envBin
        }
    }

    $initdb = Get-Command initdb -ErrorAction SilentlyContinue
    $pgctl = Get-Command pg_ctl -ErrorAction SilentlyContinue
    if ($initdb -and $pgctl) {
        return (Split-Path -Parent $initdb.Source)
    }

    $roots = @(
        "C:\Program Files\PostgreSQL",
        "C:\Program Files (x86)\PostgreSQL"
    )
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) {
            continue
        }
        $versions = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
        foreach ($versionDir in $versions) {
            $bin = Join-Path $versionDir.FullName "bin"
            if (
                (Test-Path (Join-Path $bin "initdb.exe")) -and
                (Test-Path (Join-Path $bin "pg_ctl.exe"))
            ) {
                return $bin
            }
        }
    }

    return $null
}

function Find-MemuraiServer {
    if ($env:MEMURAI_EXE -and (Test-Path $env:MEMURAI_EXE)) {
        return $env:MEMURAI_EXE
    }

    $wellKnown = "C:\Program Files\Memurai\memurai.exe"
    if (Test-Path $wellKnown) { return $wellKnown }

    return $null
}

function Test-DockerRuntimeAvailable {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        return $false
    }

    try {
        docker info *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Test-RedisRuntimeAvailable {
    if (Test-DockerRuntimeAvailable) { return $true }
    if (Find-MemuraiServer) { return $true }
    if (Find-RedisServer) { return $true }
    foreach ($svcName in @("Redis", "Memurai")) {
        try {
            $svc = Get-Service -Name $svcName -ErrorAction SilentlyContinue
            if ($svc) { return $true }
        } catch {
        }
    }
    return $false
}

function Ensure-RedisRuntime {
    if (Test-RedisRuntimeAvailable) {
        return $true
    }
    Write-Host "Redis runtime missing; invoking setup redis bootstrap..." -ForegroundColor Cyan
    try {
        & .\scripts\infra\setup.ps1 -RedisOnly
        return (Test-RedisRuntimeAvailable)
    } catch {
        return $false
    }
}

function Test-RedisPing {
    param(
        [string]$RedisHost,
        [int]$RedisPort
    )

    $client = $null
    $stream = $null
    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $client.ReceiveTimeout = 500
        $client.SendTimeout = 500
        $client.Connect($RedisHost, $RedisPort)
        $stream = $client.GetStream()
        $payload = [System.Text.Encoding]::ASCII.GetBytes("*1`r`n`$4`r`nPING`r`n")
        $stream.Write($payload, 0, $payload.Length)
        $buffer = New-Object byte[] 64
        $bytesRead = $stream.Read($buffer, 0, $buffer.Length)
        if ($bytesRead -le 0) { return $false }
        $response = [System.Text.Encoding]::ASCII.GetString($buffer, 0, $bytesRead)
        return $response.Contains("+PONG")
    } catch {
        return $false
    } finally {
        if ($stream) { $stream.Dispose() }
        if ($client) { $client.Dispose() }
    }
}

function Get-RedisVersion {
    param(
        [string]$RedisHost,
        [int]$RedisPort
    )

    $client = $null
    $stream = $null
    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $client.ReceiveTimeout = 2000
        $client.SendTimeout = 2000
        $client.Connect($RedisHost, $RedisPort)
        $stream = $client.GetStream()
        # Send: INFO server
        $payload = [System.Text.Encoding]::ASCII.GetBytes("*2`r`n`$4`r`nINFO`r`n`$6`r`nserver`r`n")
        $stream.Write($payload, 0, $payload.Length)
        $buffer = New-Object byte[] 4096
        $bytesRead = $stream.Read($buffer, 0, $buffer.Length)
        if ($bytesRead -le 0) { return "" }
        $response = [System.Text.Encoding]::ASCII.GetString($buffer, 0, $bytesRead)
        foreach ($line in $response -split "`r`n|`n") {
            if ($line.StartsWith("redis_version:")) {
                return $line.Substring("redis_version:".Length).Trim()
            }
        }
        return ""
    } catch {
        return ""
    } finally {
        if ($stream) { $stream.Dispose() }
        if ($client) { $client.Dispose() }
    }
}

function Test-RedisVersionOk {
    <#
    .SYNOPSIS
    Check if Redis version is >= 5.0 (required for Streams).
    Returns $true if version is OK, $false if too old, $null if version unknown.
    #>
    param(
        [string]$RedisHost,
        [int]$RedisPort
    )

    $version = Get-RedisVersion -RedisHost $RedisHost -RedisPort $RedisPort
    if (-not $version) { return $null }

    $parts = $version.Split(".")
    if ($parts.Count -lt 1) { return $null }
    try {
        $major = [int]$parts[0]
        return ($major -ge 5)
    } catch {
        return $null
    }
}

function Send-RedisShutdown {
    param(
        [string]$RedisHost,
        [int]$RedisPort
    )

    $client = $null
    $stream = $null
    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $client.ReceiveTimeout = 500
        $client.SendTimeout = 500
        $client.Connect($RedisHost, $RedisPort)
        $stream = $client.GetStream()
        $payload = [System.Text.Encoding]::ASCII.GetBytes("*2`r`n`$8`r`nSHUTDOWN`r`n`$6`r`nNOSAVE`r`n")
        $stream.Write($payload, 0, $payload.Length)
    } catch {
    } finally {
        if ($stream) { $stream.Dispose() }
        if ($client) { $client.Dispose() }
    }
}

function Start-RedisDocker {
    param(
        [string]$RedisHost,
        [int]$RedisPort,
        [string]$ContainerName,
        [string]$Image
    )

    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        return $false
    }
    try {
        docker info *> $null
        if ($LASTEXITCODE -ne 0) { return $false }
    } catch {
        return $false
    }

    try {
        docker container inspect $ContainerName *> $null
        if ($LASTEXITCODE -eq 0) {
            docker start $ContainerName *> $null
            return ($LASTEXITCODE -eq 0)
        }
    } catch {
    }

    try {
        docker run --name $ContainerName --detach --publish "${RedisHost}:${RedisPort}:6379" $Image redis-server --save "" --appendonly no *> $null
        if ($LASTEXITCODE -eq 0) {
            $script:redisDockerCreatedByScript = $true
            return $true
        }
    } catch {
    }

    return $false
}

function Start-RedisLocal {
    param(
        [string]$RedisHost,
        [int]$RedisPort
    )

    # Prefer Memurai service first - it supports Redis 5+ features (Streams)
    # The old Redis.Redis/redis-64 packages install Redis 3.0 which does NOT
    # support Streams and will silently break trade signal streaming.
    foreach ($svcName in @("Memurai", "Redis")) {
        try {
            $svc = Get-Service -Name $svcName -ErrorAction SilentlyContinue
            if ($svc) {
                if ($svc.Status -ne "Running") {
                    Start-Service -Name $svcName
                }
                return $true
            }
        } catch {
        }
    }

    $memuraiPath = Find-MemuraiServer
    if ($memuraiPath) {
        try {
            Start-Process -FilePath $memuraiPath -ArgumentList @("--bind", $RedisHost, "--port", "$RedisPort") -WindowStyle Hidden | Out-Null
            return $true
        } catch {
        }
    }

    $redisServerPath = Find-RedisServer
    if ($redisServerPath) {
        try {
            Start-Process -FilePath $redisServerPath -ArgumentList "--bind $RedisHost --port $RedisPort --save `"`" --appendonly no" -WindowStyle Hidden | Out-Null
            return $true
        } catch {
        }
    }

    return $false
}

function Warn-RedisVersionIfOld {
    param(
        [string]$RedisHost,
        [int]$RedisPort
    )

    $version = Get-RedisVersion -RedisHost $RedisHost -RedisPort $RedisPort
    if (-not $version) { return }

    $versionOk = Test-RedisVersionOk -RedisHost $RedisHost -RedisPort $RedisPort
    if ($versionOk -eq $false) {
        Write-Host ""
        Write-Host "WARNING: Redis version $version does NOT support Streams (requires >= 5.0)." -ForegroundColor Red
        Write-Host "Trade signal streaming will be DISABLED. The bot will fall back to slower DB polling." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "To fix, install a modern Redis:" -ForegroundColor Cyan
        Write-Host "  Option 1: Docker Desktop  ->  automatically uses redis:7-alpine" -ForegroundColor White
        Write-Host "  Option 2: Memurai         ->  winget install Memurai.MemuraiDeveloper" -ForegroundColor White
        Write-Host "  Option 3: WSL2            ->  wsl --install && sudo apt install redis-server" -ForegroundColor White
        Write-Host ""
    } elseif ($versionOk -eq $true) {
        Write-Host "Redis version $version (Streams supported)" -ForegroundColor Green
    }
}

function Ensure-Redis {
    param(
        [string]$RedisHost,
        [int]$RedisPort,
        [string]$ContainerName,
        [string]$Image
    )

    if (Test-RedisPing -RedisHost $RedisHost -RedisPort $RedisPort) {
        $versionOk = Test-RedisVersionOk -RedisHost $RedisHost -RedisPort $RedisPort
        if ($versionOk -eq $false) {
            # Running Redis is too old. Try Docker for a modern version first.
            Write-Host "Running Redis is too old for Streams support. Attempting Docker upgrade..." -ForegroundColor Yellow
            $dockerStarted = Start-RedisDocker -RedisHost $RedisHost -RedisPort $RedisPort -ContainerName $ContainerName -Image $Image
            if (-not $dockerStarted) {
                # Can't replace it - warn and continue with degraded mode
                Warn-RedisVersionIfOld -RedisHost $RedisHost -RedisPort $RedisPort
                return
            }
            if (Wait-ForService -TargetHost $RedisHost -Port $RedisPort) {
                $script:redisStartedByScript = $true
                $script:redisStartMode = "docker"
                Warn-RedisVersionIfOld -RedisHost $RedisHost -RedisPort $RedisPort
                return
            }
            # Docker started but can't reach it - fall through, old Redis still works
            Warn-RedisVersionIfOld -RedisHost $RedisHost -RedisPort $RedisPort
            return
        }
        Write-Host "Redis already running on ${RedisHost}:${RedisPort}" -ForegroundColor Green
        Warn-RedisVersionIfOld -RedisHost $RedisHost -RedisPort $RedisPort
        return
    }

    if (-not (Ensure-RedisRuntime)) {
        Write-Host "Failed to provision Redis runtime automatically." -ForegroundColor Red
        Write-Host "Install Docker Desktop, Memurai, or redis-server, then rerun." -ForegroundColor Yellow
        exit 1
    }

    Write-Host "Starting Redis..." -ForegroundColor Cyan

    # Prefer Docker (redis:7-alpine) - guaranteed modern Redis with Streams support
    $dockerStarted = Start-RedisDocker -RedisHost $RedisHost -RedisPort $RedisPort -ContainerName $ContainerName -Image $Image
    if ($dockerStarted -and (Wait-ForService -TargetHost $RedisHost -Port $RedisPort)) {
        $script:redisStartedByScript = $true
        $script:redisStartMode = "docker"
        Write-Host "Redis started via Docker on ${RedisHost}:${RedisPort}" -ForegroundColor Green
        Warn-RedisVersionIfOld -RedisHost $RedisHost -RedisPort $RedisPort
        return
    }

    # Fall back to local (Memurai service preferred over redis-server.exe)
    $localStarted = Start-RedisLocal -RedisHost $RedisHost -RedisPort $RedisPort
    if ($localStarted -and (Wait-ForService -TargetHost $RedisHost -Port $RedisPort)) {
        $script:redisStartedByScript = $true
        $script:redisStartMode = "local"
        Write-Host "Redis started via local service on ${RedisHost}:${RedisPort}" -ForegroundColor Green
        Warn-RedisVersionIfOld -RedisHost $RedisHost -RedisPort $RedisPort
        return
    }

    Write-Host "Failed to start Redis automatically." -ForegroundColor Red
    Write-Host "Install Docker Desktop, Memurai, or redis-server, then rerun." -ForegroundColor Yellow
    exit 1
}

function Cleanup-StartedRedis {
    if (-not $script:redisStartedByScript) {
        return
    }

    if ($script:redisStartMode -eq "docker") {
        if (Get-Command docker -ErrorAction SilentlyContinue) {
            try { docker stop $redisContainerName *> $null } catch {}
            if ($script:redisDockerCreatedByScript) {
                try { docker rm $redisContainerName *> $null } catch {}
            }
        }
        return
    }

    if ($script:redisStartMode -eq "local") {
        # If we started via a Windows service, stop it gracefully
        foreach ($svcName in @("Redis", "Memurai")) {
            try {
                $svc = Get-Service -Name $svcName -ErrorAction SilentlyContinue
                if ($svc -and $svc.Status -eq "Running") {
                    Stop-Service -Name $svcName -Force -ErrorAction SilentlyContinue
                    return
                }
            } catch {}
        }
        # Otherwise send raw SHUTDOWN to the redis-server process
        if (Test-RedisPing -RedisHost $redisHost -RedisPort $redisPort) {
            Send-RedisShutdown -RedisHost $redisHost -RedisPort $redisPort
        }
    }
}

function Test-PostgresRuntimeAvailable {
    if (Test-DockerRuntimeAvailable) { return $true }
    return [bool](Find-PostgresBinDir)
}

function Ensure-PostgresRuntime {
    if (Test-PostgresRuntimeAvailable) {
        return $true
    }

    Write-Host "Postgres runtime missing; invoking setup postgres bootstrap..." -ForegroundColor Cyan
    try {
        & .\scripts\infra\setup.ps1 -PostgresOnly
        return (Test-PostgresRuntimeAvailable)
    } catch {
        return $false
    }
}

function Start-PostgresDocker {
    param(
        [string]$PgHost,
        [int]$Port,
        [string]$Db,
        [string]$User,
        [string]$Password,
        [string]$ContainerName,
        [string]$Image,
        [string]$DataDir
    )

    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        return $false
    }

    try {
        docker info *> $null
        if ($LASTEXITCODE -ne 0) { return $false }
    } catch {
        return $false
    }

    try {
        docker container inspect $ContainerName *> $null
        if ($LASTEXITCODE -eq 0) {
            docker start $ContainerName *> $null
            return ($LASTEXITCODE -eq 0)
        }
    } catch {
    }

    try {
        New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
        docker run --name $ContainerName --detach --publish "${PgHost}:${Port}:5432" --env "POSTGRES_DB=$Db" --env "POSTGRES_USER=$User" --env "POSTGRES_PASSWORD=$Password" --volume "${DataDir}:/var/lib/postgresql/data" $Image *> $null
        if ($LASTEXITCODE -eq 0) {
            $script:postgresDockerCreatedByScript = $true
            return $true
        }
    } catch {
    }

    return $false
}

function Start-PostgresLocal {
    param(
        [string]$PgHost,
        [int]$Port,
        [string]$User,
        [string]$DataDir
    )

    $binDir = Find-PostgresBinDir
    if (-not $binDir) {
        return $false
    }

    $initdbPath = Join-Path $binDir "initdb.exe"
    $pgctlPath = Join-Path $binDir "pg_ctl.exe"
    if (-not (Test-Path $initdbPath) -or -not (Test-Path $pgctlPath)) {
        return $false
    }

    try {
        New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    } catch {
        return $false
    }

    $pgVersionPath = Join-Path $DataDir "PG_VERSION"
    if (-not (Test-Path $pgVersionPath)) {
        try {
            & $initdbPath -D $DataDir -U $User --encoding=UTF8 --locale=C *> $null
            if ($LASTEXITCODE -ne 0) { return $false }

            @"
local all all trust
host all all 127.0.0.1/32 trust
host all all ::1/128 trust
"@ | Set-Content -Path (Join-Path $DataDir "pg_hba.conf") -Encoding UTF8

            Add-Content -Path (Join-Path $DataDir "postgresql.conf") -Value "listen_addresses = '$PgHost'"
            Add-Content -Path (Join-Path $DataDir "postgresql.conf") -Value "port = $Port"
        } catch {
            return $false
        }
    }

    # Start via Start-Process so pg_ctl doesn't block the PowerShell pipeline.
    # The caller's Wait-ForService handles readiness polling.
    # On Windows a hard-killed Postgres can leave stale shared-memory
    # that causes the first start to fail immediately; we retry once.
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        try {
            $proc = Start-Process -FilePath $pgctlPath -ArgumentList "-D `"$DataDir`" -o `"-h $PgHost -p $Port`" start" -WindowStyle Hidden -Wait:$false -PassThru
            Start-Sleep -Milliseconds 500
            if (-not $proc.HasExited -or $proc.ExitCode -eq 0) { return $true }
        } catch {}

        if ($attempt -eq 1) {
            # Stale shared-memory from a crashed process is the most common
            # reason for failure.  Remove the PID file and pause to let the
            # OS reclaim the segment before the second attempt.
            $stalePid = Join-Path $DataDir "postmaster.pid"
            Remove-Item -Path $stalePid -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 3
        }
    }
    return $false
}

function Ensure-PostgresFirewallRule {
    <#
    .SYNOPSIS
    Ensure an inbound firewall allow rule exists for postgres.exe.

    On Windows 11, when postgres.exe first listens on a port, the firewall may
    silently block inbound connections if the UAC allow dialog was not shown or
    was dismissed (common when started from a background script).  This function
    adds an explicit allow rule for the binary.  Requires elevation; fails
    silently without it.
    #>
    param([string]$PostgresBinDir)

    $pgExe = Join-Path $PostgresBinDir "postgres.exe"
    if (-not (Test-Path $pgExe)) { return }

    $ruleName = "Homerun - PostgreSQL Server"

    # Check if rule already exists
    $existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    if ($existing) { return }

    try {
        New-NetFirewallRule `
            -DisplayName $ruleName `
            -Direction Inbound `
            -Action Allow `
            -Protocol TCP `
            -Program $pgExe `
            -Profile Any `
            -ErrorAction Stop | Out-Null
    } catch {
        # Not elevated - fall back to netsh which sometimes works or just warn
        try {
            netsh advfirewall firewall add rule name="$ruleName" dir=in action=allow protocol=TCP program="$pgExe" profile=any *> $null
        } catch {
            Write-Host "Warning: Could not add firewall rule for PostgreSQL. If connections fail, run as Administrator once or manually allow '$pgExe' in Windows Firewall." -ForegroundColor Yellow
        }
    }
}

function Test-PostgresDockerListenerOwned {
    param(
        [string]$ContainerName,
        [int]$Port
    )

    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        return $false
    }

    try {
        docker container inspect $ContainerName *> $null
        if ($LASTEXITCODE -ne 0) { return $false }
    } catch {
        return $false
    }

    try {
        $running = docker inspect -f "{{.State.Running}}" $ContainerName 2>$null
        if (($running | Out-String).Trim().ToLowerInvariant() -ne "true") {
            return $false
        }
        $hostPort = docker inspect -f "{{with index .NetworkSettings.Ports \"5432/tcp\"}}{{(index . 0).HostPort}}{{end}}" $ContainerName 2>$null
        return ((($hostPort | Out-String).Trim()) -eq "$Port")
    } catch {
        return $false
    }
}

function Test-LocalPostgresListenerOwned {
    param(
        [string]$DataDir,
        [int]$Port
    )

    $pidPath = Join-Path $DataDir "postmaster.pid"
    if (-not (Test-Path $pidPath)) {
        return $false
    }

    try {
        $lines = Get-Content -Path $pidPath -ErrorAction Stop
        if ($lines.Count -lt 4) { return $false }
        $pidValue = ($lines[0] | Out-String).Trim()
        $portValue = ($lines[3] | Out-String).Trim()
        if (-not $pidValue -or -not $portValue) { return $false }
        $proc = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
        if (-not $proc) { return $false }
        return ($portValue -eq "$Port")
    } catch {
        return $false
    }
}

function Test-LauncherPostgresListenerOwned {
    param(
        [string]$ContainerName,
        [string]$DataDir,
        [int]$Port
    )

    if (Test-PostgresDockerListenerOwned -ContainerName $ContainerName -Port $Port) {
        return $true
    }
    return (Test-LocalPostgresListenerOwned -DataDir $DataDir -Port $Port)
}

function Get-AvailablePostgresPort {
    param(
        [string]$PgHost,
        [int]$StartPort
    )

    for ($port = $StartPort; $port -le ($StartPort + 32); $port++) {
        if (-not (Test-TcpPort -TargetHost $PgHost -Port $port)) {
            return $port
        }
    }
    return $null
}

function Get-RunningLocalPostgresPort {
    param(
        [string]$DataDir
    )

    $pidPath = Join-Path $DataDir "postmaster.pid"
    if (-not (Test-Path $pidPath)) {
        return $null
    }

    try {
        $lines = Get-Content -Path $pidPath -ErrorAction Stop
        if ($lines.Count -lt 4) { return $null }
        $pidValue = ($lines[0] | Out-String).Trim()
        $portValue = ($lines[3] | Out-String).Trim()
        if (-not $pidValue -or -not $portValue) { return $null }
        $proc = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
        if (-not $proc) {
            # Stale postmaster.pid from crashed process - clean it up
            Remove-Item -Path $pidPath -Force -ErrorAction SilentlyContinue
            return $null
        }
        return [int]$portValue
    } catch {
        return $null
    }
}

function Ensure-Postgres {
    param(
        [string]$PgHost,
        [int]$Port,
        [string]$Db,
        [string]$User,
        [string]$Password,
        [string]$ContainerName,
        [string]$Image,
        [string]$DataDir
    )

    # Check if our local Postgres is already running (on any port)
    $runningPort = Get-RunningLocalPostgresPort -DataDir $DataDir
    if ($runningPort) {
        $script:postgresPort = [int]$runningPort
        # Ensure firewall rule exists even for already-running instances
        $pgBinDir = Find-PostgresBinDir
        if ($pgBinDir) { Ensure-PostgresFirewallRule -PostgresBinDir $pgBinDir }
        Write-Host "Postgres already running on ${PgHost}:${runningPort}" -ForegroundColor Green
        return
    }

    # Check if our Docker container is already running on the requested port
    if (Test-TcpPort -TargetHost $PgHost -Port $Port) {
        if (Test-PostgresDockerListenerOwned -ContainerName $ContainerName -Port $Port) {
            Write-Host "Postgres already running on ${PgHost}:${Port}" -ForegroundColor Green
            return
        }

        $alternatePort = Get-AvailablePostgresPort -PgHost $PgHost -StartPort ($Port + 1)
        if (-not $alternatePort) {
            Write-Host "Port ${Port} is occupied by a non-launcher service and no alternate Postgres port is available." -ForegroundColor Red
            Write-Host "Set DATABASE_URL manually or free a local port, then rerun." -ForegroundColor Yellow
            exit 1
        }

        Write-Host "Port ${Port} is in use by a non-launcher service. Launching project Postgres on ${alternatePort} instead." -ForegroundColor Yellow
        $Port = [int]$alternatePort
        $script:postgresPort = [int]$alternatePort
    }

    if (-not (Ensure-PostgresRuntime)) {
        Write-Host "Failed to provision Postgres runtime automatically." -ForegroundColor Red
        Write-Host "Install Docker Desktop or PostgreSQL tools (initdb + pg_ctl), then rerun." -ForegroundColor Yellow
        exit 1
    }

    Write-Host "Starting Postgres..." -ForegroundColor Cyan

    # Ensure a firewall allow rule exists for postgres.exe BEFORE starting.
    # On Windows 11, the firewall can silently block the binary on loopback
    # if the UAC allow dialog was never shown or was dismissed.
    $pgBinDir = Find-PostgresBinDir
    if ($pgBinDir) {
        Ensure-PostgresFirewallRule -PostgresBinDir $pgBinDir
    }

    $dockerStarted = Start-PostgresDocker -PgHost $PgHost -Port $Port -Db $Db -User $User -Password $Password -ContainerName $ContainerName -Image $Image -DataDir $DataDir
    if ($dockerStarted -and (Wait-ForService -TargetHost $PgHost -Port $Port)) {
        $script:postgresStartedByScript = $true
        $script:postgresStartMode = "docker"
        Write-Host "Postgres started via Docker on ${PgHost}:${Port}" -ForegroundColor Green
        return
    }

    $localStarted = Start-PostgresLocal -PgHost $PgHost -Port $Port -User $User -DataDir $DataDir
    if ($localStarted -and (Wait-ForService -TargetHost $PgHost -Port $Port)) {
        $script:postgresStartedByScript = $true
        $script:postgresStartMode = "local"
        Write-Host "Postgres started via local postgres on ${PgHost}:${Port}" -ForegroundColor Green
        return
    }

    Write-Host "Failed to start Postgres automatically." -ForegroundColor Red
    Write-Host "Install Docker Desktop or PostgreSQL tools (initdb + pg_ctl), then rerun." -ForegroundColor Yellow
    exit 1
}

function Cleanup-StartedPostgres {
    if (-not $script:postgresStartedByScript) {
        return
    }

    if ($script:postgresStartMode -eq "docker") {
        if (Get-Command docker -ErrorAction SilentlyContinue) {
            try { docker stop $postgresContainerName *> $null } catch {}
            if ($script:postgresDockerCreatedByScript) {
                try { docker rm $postgresContainerName *> $null } catch {}
            }
        }
        return
    }

    if ($script:postgresStartMode -eq "local") {
        $binDir = Find-PostgresBinDir
        if ($binDir) {
            $pgctlPath = Join-Path $binDir "pg_ctl.exe"
            if (Test-Path $pgctlPath) {
                try { & $pgctlPath -D $postgresDataDir -m fast -w stop *> $null } catch {}
            }
        }
    }
}

function Cleanup-StaleHomerunProcesses {
    <#
    .SYNOPSIS
    Kill orphaned Python worker processes from a previous crashed run.

    Finds python.exe processes whose command line contains "workers.runner"
    or "uvicorn" running from this project's backend directory and kills them.
    This prevents stale connections from saturating Postgres/Redis and blocking
    startup after an unclean exit.
    #>
    $projectRoot = (Get-Location).Path
    $backendDir = Join-Path $projectRoot "backend"

    try {
        $pythonProcesses = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue
    } catch {
        return
    }

    if (-not $pythonProcesses) {
        return
    }

    $killed = 0
    foreach ($proc in $pythonProcesses) {
        $cmdLine = $proc.CommandLine
        if (-not $cmdLine) { continue }

        # Only kill homerun-related processes (workers, uvicorn backend, tui)
        $isHomerun = $false
        if ($cmdLine -match "workers\.runner") { $isHomerun = $true }
        elseif ($cmdLine -match "workers\.\w+_worker") { $isHomerun = $true }
        elseif (($cmdLine -match "uvicorn") -and ($cmdLine -match "main:app")) { $isHomerun = $true }
        elseif ($cmdLine -match "tui\.py") { $isHomerun = $true }

        if (-not $isHomerun) { continue }

        # Verify the process is running from this project (not another instance)
        if ($cmdLine -notmatch [regex]::Escape($backendDir) -and
            $cmdLine -notmatch [regex]::Escape($projectRoot)) {
            continue
        }

        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
            $killed++
        } catch {}
    }

    if ($killed -gt 0) {
        Write-Host "Cleaned up $killed stale Homerun process(es) from a previous run." -ForegroundColor Yellow
        Start-Sleep -Seconds 1
    }
}

function Cleanup-LocalPostgresIfOwned {
    <#
    .SYNOPSIS
    Stop the launcher-managed local Postgres if it's running from our data directory.
    Called during shutdown even if the launcher didn't start it this session
    (i.e. it was already running when we launched).
    #>
    $binDir = Find-PostgresBinDir
    if (-not $binDir) { return }
    $pgctlPath = Join-Path $binDir "pg_ctl.exe"
    if (-not (Test-Path $pgctlPath)) { return }

    if (Test-LocalPostgresListenerOwned -DataDir $postgresDataDir -Port $script:postgresPort) {
        try { & $pgctlPath -D $postgresDataDir -m fast -w stop *> $null } catch {}
    }
}

function Cleanup-LocalRedisIfOwned {
    <#
    .SYNOPSIS
    Stop the launcher-managed local Redis if it was started as a standalone process.
    #>
    if (Test-RedisPing -RedisHost $redisHost -RedisPort $redisPort) {
        Send-RedisShutdown -RedisHost $redisHost -RedisPort $redisPort
    }
}

function Test-NeedsSetup {
    if (-not (Test-Path "backend\venv")) { return $true }
    if (-not (Test-Path "backend\venv\Scripts\python.exe")) { return $true }
    if (-not (Test-Path "frontend\node_modules")) { return $true }
    if (-not (Test-Path ".setup-stamp.json")) { return $true }

    $venvPython = "backend\venv\Scripts\python.exe"
    try {
        & $venvPython -c "import sys; raise SystemExit(0 if sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 13 else 1)" *> $null
        if ($LASTEXITCODE -ne 0) { return $true }
    } catch {
        return $true
    }
    try {
        & $venvPython -c "import py_clob_client, eth_account" *> $null
        if ($LASTEXITCODE -ne 0) { return $true }
    } catch {
        return $true
    }

    try {
        $stamp = Get-Content ".setup-stamp.json" -Raw | ConvertFrom-Json
    } catch {
        return $true
    }

    function Get-HashOrMissing {
        param([string]$Path)
        if (-not (Test-Path $Path)) { return "missing" }
        return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }

    $fingerprintPython = "backend\venv\Scripts\python.exe"

    $pythonVersion = (& $fingerprintPython -c "import platform; print(platform.python_version())")
    if ($stamp.python_version -ne $pythonVersion) { return $true }
    if ($stamp.requirements_sha256 -ne (Get-HashOrMissing "backend\requirements.txt")) { return $true }
    if ($stamp.requirements_trading_sha256 -ne (Get-HashOrMissing "backend\requirements-trading.txt")) { return $true }
    if ($stamp.package_json_sha256 -ne (Get-HashOrMissing "frontend\package.json")) { return $true }
    if ($stamp.package_lock_sha256 -ne (Get-HashOrMissing "frontend\package-lock.json")) { return $true }
    if ($stamp.launcher_tools_package_json_sha256 -ne (Get-HashOrMissing "scripts\infra\tooling\package.json")) { return $true }
    if ($stamp.launcher_tools_package_lock_sha256 -ne (Get-HashOrMissing "scripts\infra\tooling\package-lock.json")) { return $true }

    return $false
}

if (Test-NeedsSetup) {
    Write-Host "Setup missing or stale. Running setup..." -ForegroundColor Yellow
    & .\scripts\infra\setup.ps1
}

# Kill orphaned workers from a previous crashed run before starting services.
# Stale processes hold Postgres/Redis connections that can block startup.
Cleanup-StaleHomerunProcesses

# The Npcap Loopback Adapter (Wireshark/Nmap) can silently break loopback
# TCP connections, causing Postgres to appear to listen but drop all traffic.
Test-NpcapLoopbackInterference

try {
    Ensure-Redis -RedisHost $redisHost -RedisPort $redisPort -ContainerName $redisContainerName -Image $redisImage
    if ($env:DATABASE_URL) {
        Write-Host "Using provided DATABASE_URL; skipping launcher-managed Postgres startup." -ForegroundColor Cyan
    } else {
        Ensure-Postgres -PgHost $postgresHost -Port $postgresPort -Db $postgresDb -User $postgresUser -Password $postgresPassword -ContainerName $postgresContainerName -Image $postgresImage -DataDir $postgresDataDir
        $env:DATABASE_URL = "postgresql+asyncpg://${postgresUser}:${postgresPassword}@${postgresHost}:${script:postgresPort}/${postgresDb}"
    }

    New-Item -ItemType Directory -Path "backend\.runtime" -Force | Out-Null
    Set-Content -Path "backend\.runtime\database_url" -Value $env:DATABASE_URL -Encoding UTF8

    & backend\venv\Scripts\python.exe .\scripts\infra\ensure_postgres_ready.py --database-url $env:DATABASE_URL
    if ($LASTEXITCODE -ne 0) {
        throw "Postgres readiness validation failed"
    }

    # Activate venv
    & backend\venv\Scripts\Activate.ps1

    # Ensure TUI dependencies are installed
    try {
        python -c "import textual" 2>$null
    } catch {
        Write-Host "Installing TUI dependencies..." -ForegroundColor Cyan
        pip install -q textual rich
    }

    if ($runServiceSmokeTest) {
        python .\scripts\infra\launcher_smoke.py
        exit $LASTEXITCODE
    }

    # Launch the TUI
    python tui.py @tuiArgs
} finally {
    # Kill any remaining Homerun Python processes (workers, backend, etc.)
    Cleanup-StaleHomerunProcesses

    # Stop launcher-managed services
    Cleanup-StartedPostgres
    Cleanup-StartedRedis

    # Also stop services the launcher adopted (already running when we started)
    if (-not $script:databaseUrlWasProvided) {
        Cleanup-LocalPostgresIfOwned
    }
    Cleanup-LocalRedisIfOwned
}
