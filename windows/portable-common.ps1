Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-TrellisRoot {
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
    if (-not (Test-Path -LiteralPath (Join-Path $candidate 'app.py'))) {
        throw "Could not locate the TRELLIS.2 application root."
    }
    return $candidate
}

function Initialize-PortableEnvironment {
    param([switch]$Create)

    $root = Get-TrellisRoot
    $portable = Join-Path $root '.portable'
    $paths = [ordered]@{
        Root = $root
        Portable = $portable
        Runtime = Join-Path $portable 'runtime'
        Python = Join-Path $portable 'runtime\python\python.exe'
        Downloads = Join-Path $portable 'downloads'
        Wheels = Join-Path $portable 'downloads\wheels'
        Cache = Join-Path $portable 'cache'
        Hf = Join-Path $portable 'models\huggingface'
        Temp = Join-Path $portable 'temp'
        Sessions = Join-Path $portable 'sessions'
        Outputs = Join-Path $portable 'outputs'
        Logs = Join-Path $portable 'logs'
        Run = Join-Path $portable 'run'
        Secrets = Join-Path $portable 'secrets'
        Token = Join-Path $portable 'secrets\huggingface-token'
        Browser = Join-Path $portable 'browser-profile'
    }

    foreach ($value in $paths.Values) {
        $full = [System.IO.Path]::GetFullPath($value)
        if (-not $full.StartsWith($root + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -and $full -ne $root) {
            throw "Portable path escaped the application root: $full"
        }
    }

    if ($Create) {
        @('Runtime','Downloads','Wheels','Cache','Hf','Temp','Sessions','Outputs','Logs','Run','Secrets','Browser') |
            ForEach-Object { New-Item -ItemType Directory -Force -Path $paths[$_] | Out-Null }
    }

    $env:OPENCV_IO_ENABLE_OPENEXR = '1'
    $env:CUDA_MODULE_LOADING = 'LAZY'
    $env:ATTN_BACKEND = 'xformers'
    $env:SPARSE_ATTN_BACKEND = 'xformers'
    $env:SPARSE_CONV_BACKEND = 'flex_gemm'
    $env:HF_HOME = $paths.Hf
    $env:HF_HUB_CACHE = Join-Path $paths.Hf 'hub'
    $env:HF_TOKEN_PATH = $paths.Token
    $env:TORCH_HOME = Join-Path $paths.Cache 'torch'
    $env:TORCH_EXTENSIONS_DIR = Join-Path $paths.Cache 'torch_extensions'
    $env:TRITON_CACHE_DIR = Join-Path $paths.Cache 'triton'
    $env:CUDA_CACHE_PATH = Join-Path $paths.Cache 'cuda'
    $env:GRADIO_TEMP_DIR = Join-Path $paths.Cache 'gradio'
    $env:PIP_CACHE_DIR = Join-Path $paths.Cache 'pip'
    $env:XDG_CACHE_HOME = Join-Path $paths.Cache 'xdg'
    $env:MPLCONFIGDIR = Join-Path $paths.Cache 'matplotlib'
    $env:TEMP = $paths.Temp
    $env:TMP = $paths.Temp
    $env:GRADIO_ANALYTICS_ENABLED = 'False'
    $env:HF_HUB_DISABLE_TELEMETRY = '1'
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
    $env:DO_NOT_TRACK = '1'
    $env:TRANSFORMERS_NO_ADVISORY_WARNINGS = '1'
    $env:TOKENIZERS_PARALLELISM = 'false'
    $env:PYTHONNOUSERSITE = '1'
    $env:PIP_USER = 'no'

    return [PSCustomObject]$paths
}

function Show-PortableError {
    param([string]$Message)
    try {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show($Message, 'TRELLIS.2', 'OK', 'Error') | Out-Null
    } catch {
        Write-Error $Message
    }
}

function Get-PortableBrowser {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
        "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
        "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
        "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
    )
    return $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}

function Invoke-PortableDownload {
    param(
        [Parameter(Mandatory)][string]$Uri,
        [Parameter(Mandatory)][string]$Destination,
        [Parameter(Mandatory)][string]$Sha256
    )
    if (Test-Path -LiteralPath $Destination) {
        $existing = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($existing -eq $Sha256.ToLowerInvariant()) { return }
        Remove-Item -LiteralPath $Destination -Force
    }
    Write-Host "Downloading $(Split-Path -Leaf $Destination)..."
    Invoke-WebRequest -Uri $Uri -OutFile $Destination -UseBasicParsing
    $actual = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Sha256.ToLowerInvariant()) {
        Remove-Item -LiteralPath $Destination -Force
        throw "Checksum mismatch for $(Split-Path -Leaf $Destination)."
    }
}
