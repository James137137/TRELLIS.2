. (Join-Path $PSScriptRoot 'portable-common.ps1')
$paths = Initialize-PortableEnvironment -Create
$backend = $null
$shutdownFile = $null

try {
    if (-not (Test-Path -LiteralPath $paths.Python)) {
        throw 'The portable runtime is not installed. Run Setup TRELLIS 2.bat first.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $paths.Portable 'installation.json'))) {
        throw 'Setup has not completed. Run Setup TRELLIS 2.bat first.'
    }
    if (-not (Test-Path -LiteralPath $paths.Token)) {
        throw 'The folder-local Hugging Face token is missing. Run Setup TRELLIS 2.bat again.'
    }
    $browser = Get-PortableBrowser
    if (-not $browser) { throw 'Microsoft Edge or Google Chrome was not found.' }

    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    $port = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
    $listener.Stop()

    $runId = [Guid]::NewGuid().ToString('N')
    $readyFile = Join-Path $paths.Run "$runId.ready.json"
    $shutdownFile = Join-Path $paths.Run "$runId.shutdown"
    $errorFile = Join-Path $paths.Run "$runId.error.txt"
    $stdoutLog = Join-Path $paths.Logs "$runId.stdout.log"
    $stderrLog = Join-Path $paths.Logs "$runId.stderr.log"

    $serverScript = Join-Path $paths.Root 'portable_server.py'
    $arguments = @(
        "`"$serverScript`"",
        '--host', '127.0.0.1',
        '--port', $port,
        '--ready-file', "`"$readyFile`"",
        '--shutdown-file', "`"$shutdownFile`"",
        '--error-file', "`"$errorFile`""
    ) -join ' '
    $backend = Start-Process -FilePath $paths.Python -ArgumentList $arguments -WorkingDirectory $paths.Root -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog

    $deadline = (Get-Date).AddMinutes(5)
    while (-not (Test-Path -LiteralPath $readyFile)) {
        if ($backend.HasExited) {
            $message = if (Test-Path -LiteralPath $errorFile) { Get-Content -LiteralPath $errorFile -Raw } else { Get-Content -LiteralPath $stderrLog -Raw }
            throw "TRELLIS.2 could not start.`n`n$message"
        }
        if ((Get-Date) -gt $deadline) { throw 'TRELLIS.2 did not start within five minutes.' }
        Start-Sleep -Milliseconds 250
    }

    $url = "http://127.0.0.1:$port/?__theme=dark"
    $browserArguments = @(
        "--app=$url",
        "`"--user-data-dir=$($paths.Browser)`"",
        '--no-first-run',
        '--disable-sync',
        '--disable-background-mode',
        '--disable-features=msEdgeFirstRunExperience'
    )
    $browserProcess = Start-Process -FilePath $browser -ArgumentList $browserArguments -PassThru
    Wait-Process -Id $browserProcess.Id
} catch {
    Show-PortableError -Message $_.Exception.Message
} finally {
    if ($shutdownFile) {
        [System.IO.File]::WriteAllText($shutdownFile, 'shutdown', [System.Text.Encoding]::ASCII)
    }
    if ($backend -and -not $backend.HasExited) {
        try { Wait-Process -Id $backend.Id -Timeout 20 -ErrorAction Stop } catch { Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue }
    }
}
