# Homerun - Windows Run Script (TUI)
# Run: .\scripts\run.ps1

$ErrorActionPreference = "Stop"

# Navigate to project root (parent of scripts\)
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

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

function Test-TcpPort {
    param(
        [string]$Host,
        [int]$Port
    )

    $client = $null
    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $client.ReceiveTimeout = 500
        $client.SendTimeout = 500
        $client.Connect($Host, $Port)
        return $true
    } catch {
        return $false
    } finally {
        if ($client) { $client.Dispose() }
    }
}

function Wait-ForService {
    param(
        [string]$Host,
        [int]$Port
    )

    for ($i = 0; $i -lt 40; $i++) {
        if (Test-TcpPort -Host $Host -Port $Port) {
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

function Test-RedisRuntimeAvailable {
    if (Get-Command docker -ErrorAction SilentlyContinue) { return $true }
    if (Find-RedisServer) { return $true }
    try {
        $svc = Get-Service -Name "Memurai" -ErrorAction SilentlyContinue
        if ($svc) { return $true }
    } catch {
    }
    return $false
}

function Ensure-RedisRuntime {
    if (Test-RedisRuntimeAvailable) {
        return $true
    }
    Write-Host "Redis runtime missing; invoking setup redis bootstrap..." -ForegroundColor Cyan
    try {
        & .\scripts\setup.ps1 -RedisOnly
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

    $redisServerPath = Find-RedisServer
    if ($redisServerPath) {
        try {
            Start-Process -FilePath $redisServerPath -ArgumentList @("--bind", $RedisHost, "--port", "$RedisPort", "--save", "", "--appendonly", "no") -WindowStyle Hidden | Out-Null
            return $true
        } catch {
        }
    }

    try {
        $memuraiService = Get-Service -Name "Memurai" -ErrorAction SilentlyContinue
        if ($memuraiService) {
            if ($memuraiService.Status -ne "Running") {
                Start-Service -Name "Memurai"
            }
            return $true
        }
    } catch {
    }

    return $false
}

function Ensure-Redis {
    param(
        [string]$RedisHost,
        [int]$RedisPort,
        [string]$ContainerName,
        [string]$Image
    )

    if (Test-RedisPing -RedisHost $RedisHost -RedisPort $RedisPort) {
        Write-Host "Redis already running on ${RedisHost}:${RedisPort}" -ForegroundColor Green
        return
    }

    if (-not (Ensure-RedisRuntime)) {
        Write-Host "Failed to provision Redis runtime automatically." -ForegroundColor Red
        Write-Host "Install Docker Desktop, redis-server, or Memurai, then rerun." -ForegroundColor Yellow
        exit 1
    }

    Write-Host "Starting Redis..." -ForegroundColor Cyan
    $dockerStarted = Start-RedisDocker -RedisHost $RedisHost -RedisPort $RedisPort -ContainerName $ContainerName -Image $Image
    if ($dockerStarted -and (Wait-ForService -Host $RedisHost -Port $RedisPort)) {
        $script:redisStartedByScript = $true
        $script:redisStartMode = "docker"
        Write-Host "Redis started via Docker on ${RedisHost}:${RedisPort}" -ForegroundColor Green
        return
    }

    $localStarted = Start-RedisLocal -RedisHost $RedisHost -RedisPort $RedisPort
    if ($localStarted -and (Wait-ForService -Host $RedisHost -Port $RedisPort)) {
        $script:redisStartedByScript = $true
        $script:redisStartMode = "local"
        Write-Host "Redis started via redis-server on ${RedisHost}:${RedisPort}" -ForegroundColor Green
        return
    }

    Write-Host "Failed to start Redis automatically." -ForegroundColor Red
    Write-Host "Install Docker or redis-server, then rerun." -ForegroundColor Yellow
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
        if (Test-RedisPing -RedisHost $redisHost -RedisPort $redisPort) {
            Send-RedisShutdown -RedisHost $redisHost -RedisPort $redisPort
        }
    }
}

function Test-PostgresRuntimeAvailable {
    if (Get-Command docker -ErrorAction SilentlyContinue) { return $true }
    return [bool](Find-PostgresBinDir)
}

function Ensure-PostgresRuntime {
    if (Test-PostgresRuntimeAvailable) {
        return $true
    }

    Write-Host "Postgres runtime missing; invoking setup postgres bootstrap..." -ForegroundColor Cyan
    try {
        & .\scripts\setup.ps1 -PostgresOnly
        return (Test-PostgresRuntimeAvailable)
    } catch {
        return $false
    }
}

function Start-PostgresDocker {
    param(
        [string]$Host,
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
        docker run --name $ContainerName --detach --publish "${Host}:${Port}:5432" --env "POSTGRES_DB=$Db" --env "POSTGRES_USER=$User" --env "POSTGRES_PASSWORD=$Password" --volume "${DataDir}:/var/lib/postgresql/data" $Image *> $null
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
        [string]$Host,
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
            & $initdbPath -D $DataDir -U $User *> $null
            if ($LASTEXITCODE -ne 0) { return $false }

            @"
local all all trust
host all all 127.0.0.1/32 trust
host all all ::1/128 trust
"@ | Set-Content -Path (Join-Path $DataDir "pg_hba.conf") -Encoding UTF8

            Add-Content -Path (Join-Path $DataDir "postgresql.conf") -Value "listen_addresses = '$Host'"
            Add-Content -Path (Join-Path $DataDir "postgresql.conf") -Value "port = $Port"
        } catch {
            return $false
        }
    }

    try {
        & $pgctlPath -D $DataDir -o "-h $Host -p $Port" -w start *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Ensure-Postgres {
    param(
        [string]$Host,
        [int]$Port,
        [string]$Db,
        [string]$User,
        [string]$Password,
        [string]$ContainerName,
        [string]$Image,
        [string]$DataDir
    )

    if (Test-TcpPort -Host $Host -Port $Port) {
        Write-Host "Postgres already running on ${Host}:${Port}" -ForegroundColor Green
        return
    }

    if (-not (Ensure-PostgresRuntime)) {
        Write-Host "Failed to provision Postgres runtime automatically." -ForegroundColor Red
        Write-Host "Install Docker Desktop or PostgreSQL tools (initdb + pg_ctl), then rerun." -ForegroundColor Yellow
        exit 1
    }

    Write-Host "Starting Postgres..." -ForegroundColor Cyan
    $dockerStarted = Start-PostgresDocker -Host $Host -Port $Port -Db $Db -User $User -Password $Password -ContainerName $ContainerName -Image $Image -DataDir $DataDir
    if ($dockerStarted -and (Wait-ForService -Host $Host -Port $Port)) {
        $script:postgresStartedByScript = $true
        $script:postgresStartMode = "docker"
        Write-Host "Postgres started via Docker on ${Host}:${Port}" -ForegroundColor Green
        return
    }

    $localStarted = Start-PostgresLocal -Host $Host -Port $Port -User $User -DataDir $DataDir
    if ($localStarted -and (Wait-ForService -Host $Host -Port $Port)) {
        $script:postgresStartedByScript = $true
        $script:postgresStartMode = "local"
        Write-Host "Postgres started via local postgres on ${Host}:${Port}" -ForegroundColor Green
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

function Test-NeedsSetup {
    if (-not (Test-Path "backend\venv")) { return $true }
    if (-not (Test-Path "frontend\node_modules")) { return $true }
    if (-not (Test-Path "scripts\tooling\node_modules")) { return $true }
    if (-not (Test-Path ".setup-stamp.json")) { return $true }

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

    $fingerprintPython = "python"
    if (Test-Path "backend\venv\Scripts\python.exe") {
        $fingerprintPython = "backend\venv\Scripts\python.exe"
    }

    $pythonVersion = (& $fingerprintPython -c "import platform; print(platform.python_version())")
    if ($stamp.python_version -ne $pythonVersion) { return $true }
    if ($stamp.requirements_sha256 -ne (Get-HashOrMissing "backend\requirements.txt")) { return $true }
    if ($stamp.requirements_trading_sha256 -ne (Get-HashOrMissing "backend\requirements-trading.txt")) { return $true }
    if ($stamp.package_json_sha256 -ne (Get-HashOrMissing "frontend\package.json")) { return $true }
    if ($stamp.package_lock_sha256 -ne (Get-HashOrMissing "frontend\package-lock.json")) { return $true }
    if ($stamp.launcher_tools_package_json_sha256 -ne (Get-HashOrMissing "scripts\tooling\package.json")) { return $true }
    if ($stamp.launcher_tools_package_lock_sha256 -ne (Get-HashOrMissing "scripts\tooling\package-lock.json")) { return $true }

    return $false
}

if (Test-NeedsSetup) {
    Write-Host "Setup missing or stale. Running setup..." -ForegroundColor Yellow
    & .\scripts\setup.ps1
}

try {
    Ensure-Redis -RedisHost $redisHost -RedisPort $redisPort -ContainerName $redisContainerName -Image $redisImage
    Ensure-Postgres -Host $postgresHost -Port $postgresPort -Db $postgresDb -User $postgresUser -Password $postgresPassword -ContainerName $postgresContainerName -Image $postgresImage -DataDir $postgresDataDir

    # Ensure backend uses launcher-managed Postgres if DATABASE_URL wasn't provided.
    if (-not $env:DATABASE_URL) {
        $env:DATABASE_URL = "postgresql+asyncpg://${postgresUser}:${postgresPassword}@${postgresHost}:${postgresPort}/${postgresDb}"
    }

    & backend\venv\Scripts\python.exe .\scripts\ensure_postgres_ready.py --database-url $env:DATABASE_URL
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
        python .\scripts\launcher_smoke.py
        exit $LASTEXITCODE
    }

    # Launch the TUI
    python tui.py @tuiArgs
} finally {
    Cleanup-StartedPostgres
    Cleanup-StartedRedis
}
