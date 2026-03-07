$input_str = "protocol=https`nhost=github.com`n`n"
$process = New-Object System.Diagnostics.Process
$process.StartInfo.FileName = 'git'
$process.StartInfo.Arguments = 'credential fill'
$process.StartInfo.RedirectStandardInput = $true
$process.StartInfo.RedirectStandardOutput = $true
$process.StartInfo.RedirectStandardError = $true
$process.StartInfo.UseShellExecute = $false
$process.StartInfo.CreateNoWindow = $true
$process.Start() | Out-Null
$process.StandardInput.Write($input_str)
$process.StandardInput.Close()
$output = $process.StandardOutput.ReadToEnd()
$stderr = $process.StandardError.ReadToEnd()
$process.WaitForExit()

$token = ''
foreach($line in ($output -split "`n")) {
    if ($line -match '^password=(.*)') {
        $token = $Matches[1].Trim()
    }
}

if (-not $token) {
    Write-Host "STDERR: $stderr"
    Write-Host "STDOUT: $output"
    Write-Error "No token found"
    exit 1
}

Write-Host "Token length: $($token.Length)"

$url = 'https://api.github.com/repos/braedonsaunders/homerun/actions/jobs/65980452616/logs'
$headers = @{
    Accept = 'application/vnd.github+json'
    Authorization = "token $token"
}
try {
    $resp = Invoke-WebRequest -Uri $url -Headers $headers -MaximumRedirection 10
    $resp.Content | Out-File -FilePath "ci_logs.txt" -Encoding utf8
    Write-Host "Logs saved to ci_logs.txt ($($resp.Content.Length) bytes)"
} catch {
    Write-Host "Error: $($_.Exception.Message)"
    if ($_.Exception.Response) {
        Write-Host "Status: $($_.Exception.Response.StatusCode)"
    }
}
