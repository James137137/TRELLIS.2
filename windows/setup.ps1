param(
    [switch]$SkipModelDownload,
    [switch]$SkipTokenPrompt
)

. (Join-Path $PSScriptRoot 'portable-common.ps1')
$paths = Initialize-PortableEnvironment -Create
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'portable-manifest.json') -Raw | ConvertFrom-Json

Write-Host ''
Write-Host 'TRELLIS.2 Portable Windows Setup' -ForegroundColor Cyan
Write-Host 'Everything installed by this setup remains inside the .portable folder.'
Write-Host ''

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'This portable setup supports 64-bit Windows only.'
}
if (-not [Environment]::Is64BitOperatingSystem) {
    throw 'A 64-bit Windows installation is required.'
}

$browser = Get-PortableBrowser
if (-not $browser) {
    throw 'Microsoft Edge or Google Chrome is required. Neither browser was found.'
}

$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    throw 'The NVIDIA driver was not found. Install an NVIDIA driver before using TRELLIS.2.'
}
$gpuQuery = & $nvidiaSmi.Source --query-gpu=name,memory.total --format=csv,noheader,nounits
$gpuQueryExit = $LASTEXITCODE
$gpuLine = $gpuQuery | Select-Object -First 1
if ($gpuQueryExit -ne 0 -or -not $gpuLine) { throw 'Unable to query the NVIDIA GPU.' }
$gpuParts = $gpuLine -split ','
$gpuName = $gpuParts[0].Trim()
$gpuMemory = [int]$gpuParts[1].Trim()
if ($gpuMemory -lt 23000) {
    throw "TRELLIS.2 requires approximately 24 GB of VRAM. Detected $gpuName with $gpuMemory MiB."
}
Write-Host "GPU: $gpuName ($gpuMemory MiB)" -ForegroundColor Green

$pythonZip = Join-Path $paths.Downloads $manifest.python.filename
Invoke-PortableDownload -Uri $manifest.python.url -Destination $pythonZip -Sha256 $manifest.python.sha256

$pythonDir = Split-Path -Parent $paths.Python
if (-not (Test-Path -LiteralPath $paths.Python)) {
    Write-Host 'Extracting the private Python runtime...'
    New-Item -ItemType Directory -Force -Path $pythonDir | Out-Null
    Expand-Archive -LiteralPath $pythonZip -DestinationPath $pythonDir -Force
}

# Triton compiles a tiny CUDA driver shim on first use. The embeddable Python
# distribution omits headers/import libraries, so add them from Python's
# official NuGet package without installing a compiler or changing Windows.
if (-not (Test-Path -LiteralPath (Join-Path $pythonDir 'Include\Python.h'))) {
    $pythonDevZip = Join-Path $paths.Downloads $manifest.python_dev.filename
    Invoke-PortableDownload -Uri $manifest.python_dev.url -Destination $pythonDevZip -Sha256 $manifest.python_dev.sha256
    $pythonDevStage = Join-Path $paths.Temp 'python-dev-3.11.9'
    if (Test-Path -LiteralPath $pythonDevStage) { Remove-Item -LiteralPath $pythonDevStage -Recurse -Force }
    Expand-Archive -LiteralPath $pythonDevZip -DestinationPath $pythonDevStage -Force
    Copy-Item -LiteralPath (Join-Path $pythonDevStage 'tools\include') -Destination (Join-Path $pythonDir 'Include') -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $pythonDevStage 'tools\libs') -Destination (Join-Path $pythonDir 'libs') -Recurse -Force
    Remove-Item -LiteralPath $pythonDevStage -Recurse -Force
}

$pthFile = Join-Path $pythonDir 'python311._pth'
if (-not (Test-Path -LiteralPath $pthFile)) { throw 'The portable Python path file is missing.' }
$pthLines = Get-Content -LiteralPath $pthFile
$pthLines = $pthLines | ForEach-Object { if ($_ -eq '#import site') { 'import site' } else { $_ } }
if ($pthLines -notcontains 'Lib\site-packages') {
    $pthLines = @($pthLines | Where-Object { $_ -ne 'import site' }) + @('Lib\site-packages', 'import site')
}
if ($pthLines -notcontains '..\..\..') {
    $pthLines = @($pthLines | Where-Object { $_ -ne 'import site' }) + @('..\..\..', 'import site')
}
[System.IO.File]::WriteAllLines($pthFile, $pthLines, [System.Text.Encoding]::ASCII)

$pipWheel = Join-Path $paths.Downloads $manifest.pip.filename
Invoke-PortableDownload -Uri $manifest.pip.url -Destination $pipWheel -Sha256 $manifest.pip.sha256

function Invoke-PortablePip {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PipArguments)
    & $paths.Python -m pip @PipArguments
    if ($LASTEXITCODE -ne 0) { throw "pip failed with exit code $LASTEXITCODE." }
}

$pipInstalled = & $paths.Python -c "import importlib.util; print('yes' if importlib.util.find_spec('pip') else 'no')"
if (($pipInstalled | Select-Object -Last 1) -ne 'yes') {
    $originalPth = Get-Content -LiteralPath $pthFile
    try {
        [System.IO.File]::WriteAllLines($pthFile, @($originalPth | Where-Object { $_ -ne 'import site' }) + @($pipWheel, 'import site'), [System.Text.Encoding]::ASCII)
        & $paths.Python -m pip install --no-index --no-cache-dir $pipWheel
        if ($LASTEXITCODE -ne 0) { throw 'Unable to bootstrap pip into the portable runtime.' }
    } finally {
        [System.IO.File]::WriteAllLines($pthFile, $originalPth, [System.Text.Encoding]::ASCII)
    }
}

Write-Host 'Installing the GPU runtime (this is the largest download)...' -ForegroundColor Cyan
Invoke-PortablePip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0 xformers==0.0.30
Invoke-PortablePip install --no-cache-dir triton-windows==3.3.1.post21
Invoke-PortablePip install --no-cache-dir -r (Join-Path $PSScriptRoot 'requirements-windows.txt')
Invoke-PortablePip install --no-cache-dir 'https://github.com/EasternJournalist/utils3d/archive/9a4eb15e4021b67b12c460c7057d642626897ec8.zip'

$wheelBase = "https://github.com/visualbruno/ComfyUI-Trellis2/raw/$($manifest.windows_wheels.commit)/wheels/Windows/Torch270"
$wheelPaths = @()
foreach ($wheel in $manifest.windows_wheels.items) {
    $destination = Join-Path $paths.Wheels $wheel.filename
    Invoke-PortableDownload -Uri "$wheelBase/$($wheel.filename)" -Destination $destination -Sha256 $wheel.sha256
    $wheelPaths += $destination
}
Invoke-PortablePip install --no-cache-dir --no-deps @wheelPaths

if (-not $SkipTokenPrompt -and -not (Test-Path -LiteralPath $paths.Token)) {
    Write-Host ''
    Write-Host 'DINOv3 requires a Hugging Face account with approved access.' -ForegroundColor Yellow
    Write-Host 'The access page will open in your browser. Accept the terms, create a READ token,'
    Write-Host 'then return here and paste the token. The token is stored only in this folder.'
    Start-Process -FilePath $browser -ArgumentList @(
        'https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m',
        "--user-data-dir=$($paths.Browser)",
        '--no-first-run',
        '--disable-sync',
        '--disable-background-mode'
    ) | Out-Null
    $secureToken = Read-Host 'Hugging Face READ token' -AsSecureString
    $tokenPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
    try {
        $plainToken = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPtr)
        if ([string]::IsNullOrWhiteSpace($plainToken)) { throw 'No Hugging Face token was supplied.' }
        [System.IO.File]::WriteAllText($paths.Token, $plainToken.Trim(), [System.Text.UTF8Encoding]::new($false))
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPtr)
    }
}

Write-Host 'Verifying native CUDA modules...' -ForegroundColor Cyan
& $paths.Python (Join-Path $PSScriptRoot 'verify_runtime.py')
if ($LASTEXITCODE -ne 0) { throw 'The portable runtime verification failed.' }

$modelsDownloaded = $false
if (-not $SkipModelDownload) {
    if (-not (Test-Path -LiteralPath $paths.Token)) {
        throw 'A folder-local Hugging Face token is required before model download.'
    }
    Write-Host 'Downloading TRELLIS.2, DINOv3, and background-removal weights...' -ForegroundColor Cyan
    & $paths.Python (Join-Path $PSScriptRoot 'download_models.py')
    if ($LASTEXITCODE -ne 0) { throw 'Model access or download failed. Confirm DINOv3 access and rerun setup.' }
    $modelsDownloaded = $true
}

$status = [ordered]@{
    installed_at = (Get-Date).ToString('o')
    gpu = $gpuName
    gpu_memory_mib = $gpuMemory
    python = $manifest.python.version
    torch = '2.7.0+cu128'
    xformers = '0.0.30'
    triton_windows = '3.3.1.post21'
    wheel_source_commit = $manifest.windows_wheels.commit
    models_downloaded = $modelsDownloaded
}
$status | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $paths.Portable 'installation.json') -Encoding UTF8

Write-Host ''
Write-Host 'Setup complete. Double-click Launch TRELLIS 2.bat to start.' -ForegroundColor Green
Write-Host 'Deleting this TRELLIS.2 folder removes the complete portable installation.'
