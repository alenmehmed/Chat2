# Runs the full Chat2 model-creation pipeline in one shot: uv environment setup, tokenizer build, pretrain data prep, pretraining, and chat fine-tuning.

[CmdletBinding()]
param(
    [ValidateSet('env', 'tokenizer', 'pretraindata', 'pretrain', 'train')]
    [string[]]$Force = @()
)

$Root = $PSScriptRoot
Set-Location $Root

# LOGGING
$LogDir = Join-Path $Root 'logs'
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$LogFile = Join-Path $LogDir ("pipeline_{0}.txt" -f (Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'))
New-Item -ItemType File -Path $LogFile | Out-Null

function Write-Log {
    param([string]$Message = '', [string]$Color = 'Gray')
    Write-Host $Message -ForegroundColor $Color
    Add-Content -LiteralPath $script:LogFile -Value $Message -Encoding utf8 # Appends 
}

# log only each tqdm bar's final state (the 100% line) once the bar finishes.
$ConsoleWidth = 100
try { $ConsoleWidth = [Math]::Max(40, $Host.UI.RawUI.WindowSize.Width - 1) } catch { }
$script:TqdmLast = $null

function Close-TqdmLine {
    if ($null -ne $script:TqdmLast) {
        Write-Host ""   # terminate the in-place progress line
        Add-Content -LiteralPath $script:LogFile -Value $script:TqdmLast -Encoding utf8
        $script:TqdmLast = $null
    }
}

function Write-StageLine {
    param([string]$Line)

    # tqdm's leading \r can produce empty chunks mid-bar
    if ($Line.Length -eq 0 -and $null -ne $script:TqdmLast) { return }

    if ($Line -match '%\|') {
        $text = $Line
        if ($text.Length -gt $script:ConsoleWidth) { $text = $text.Substring(0, $script:ConsoleWidth) }
        Write-Host ("`r" + $text.PadRight($script:ConsoleWidth)) -NoNewline
        $script:TqdmLast = $Line   # remember only the most recent refresh
        return
    }

    Close-TqdmLine
    Write-Host $Line
    Add-Content -LiteralPath $script:LogFile -Value $Line -Encoding utf8
}

$Summary = @()

function Invoke-Stage {
    param(
        [string]$Name,          # -Force key
        [string]$Description,   # human label
        [string]$Command,       # command line, run through cmd.exe
        [bool]$AlreadyDone,     # output already on disk
        [string]$DoneReason     # what was found, for the skip message
    )

    $forced = $Force -contains $Name

    if ($AlreadyDone -and -not $forced) {
        Write-Log ""
        Write-Log "[SKIP] $Description" 'DarkGray'
        Write-Log "       $DoneReason" 'DarkGray'
        $script:Summary += [pscustomobject]@{ Stage = $Name; Status = 'SKIPPED'; Duration = '-' }
        return
    }

    Write-Log ""
    Write-Log ("=" * 78) 'Cyan'
    if ($AlreadyDone) {
        Write-Log "[FORCE] $Description  (output exists, re-running anyway)" 'Yellow'
    } else {
        Write-Log "[RUN]   $Description" 'Cyan'
    }
    Write-Log "        > $Command" 'DarkGray'
    Write-Log ("=" * 78) 'Cyan'

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    # stderr is merged inside cmd.exe, not by PowerShell. PS 5.1 wraps native stderr
    # in ErrorRecords, which would turn every tqdm progress line red and set $? false.
    $script:TqdmLast = $null
    & cmd.exe /d /c "$Command 2>&1" | ForEach-Object { Write-StageLine $_ }
    $code = $LASTEXITCODE
    Close-TqdmLine
    $sw.Stop()

    $elapsed = $sw.Elapsed.ToString('hh\:mm\:ss')

    if ($code -ne 0) {
        Write-Log ""
        Write-Log "[FAIL]  $Description exited with code $code after $elapsed" 'Red'
        Write-Log "        Log: $script:LogFile" 'Red'
        exit $code
    }

    Write-Log "[DONE]  $Description in $elapsed" 'Green'
    $script:Summary += [pscustomobject]@{ Stage = $Name; Status = 'RAN'; Duration = $elapsed }
}

$RunTimer = [System.Diagnostics.Stopwatch]::StartNew()

Write-Log ("=" * 78) 'White'
Write-Log " Chat2 pipeline" 'White'
Write-Log " started : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" 'White'
Write-Log " root    : $Root" 'White'
Write-Log " log     : $LogFile" 'White'

if ($Force.Count -gt 0) { Write-Log " forced  : $($Force -join ', ')" 'Yellow' }
Write-Log ("=" * 78) 'White'

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Log "[FAIL]  'uv' is not on PATH." 'Red'
    exit 1
}

# stdout is block-buffered when piped, stderr is not - so print() output would
# arrive in 8 KB clumps, out of order against the tqdm bars in the stream.
$env:PYTHONUNBUFFERED = '1'

# Stage 1 - Environment 
$EnvDone = $false

$PyProject = Join-Path $Root 'pyproject.toml'
$UvLock = Join-Path $Root 'uv.lock'
$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
$SyncStamp = Join-Path $Root '.venv\.pipeline_sync_stamp'

if ((Test-Path $VenvPython) -and (Test-Path $SyncStamp)) {
    $stampTime = (Get-Item $SyncStamp).LastWriteTime
    $EnvDone = $true
    foreach ($f in @($PyProject, $UvLock)) {
        if ((Test-Path $f) -and ((Get-Item $f).LastWriteTime -gt $stampTime)) { $EnvDone = $false }
    }
}

Invoke-Stage -Name 'env' `
             -Description 'Python environment + dependencies (uv sync)' `
             -Command 'uv sync' `
             -AlreadyDone $EnvDone `
             -DoneReason "venv present and up to date with pyproject.toml / uv.lock"

if (-not $EnvDone -or ($Force -contains 'env')) {
    Set-Content -LiteralPath $SyncStamp -Value (Get-Date -Format 'o') -Encoding utf8
}

# no CPU fallback
# done after environment setup, so uv and torch are guaranteed to be importable
$cuda = & uv run python -c "import torch; print(torch.cuda.is_available())"
if ("$cuda".Trim() -eq 'True') {
    Write-Log "[INFO]  CUDA available: True" 'Green'
} else {
    Write-Log "[FAIL]  CUDA not available - pretrain and train will fail" 'Red'
    exit 1
}

# Stage 2 - Tokenizer
$Tokenizer = Join-Path $Root 'models\bpe.json'
$Corpus = Join-Path $Root 'tokenizer\corpus.txt'


Invoke-Stage -Name 'tokenizer' `
             -Description 'Build tokenizer' `
             -Command 'uv run python tokenizer/build_tokenizer.py' `
             -AlreadyDone (Test-Path $Tokenizer) `
             -DoneReason "found $Tokenizer"

# Stage 3 - Prepare pretrain data 
$PtTokens = Join-Path $Root 'models\Pretrain\pretrain_tokens.bin'
$PtMeta = Join-Path $Root 'models\Pretrain\pretrain_tokens_meta.json'

$PretrainDataDone = (Test-Path $PtTokens) -and (Test-Path $PtMeta)

if ((-not $PretrainDataDone -or ($Force -contains 'pretraindata')) -and -not (Test-Path $Corpus)) {
    Write-Log "[FAIL]  Missing corpus.txt." 'Red'
    exit 1
}

Invoke-Stage -Name 'pretraindata' `
             -Description 'Prepare packed pretrain tokens' `
             -Command 'uv run python training/pretraining/prepare_pretrain_data.py' `
             -AlreadyDone $PretrainDataDone `
             -DoneReason "Found $PtTokens and $PtMeta"

# Stage 4 - Pretraining 
$PtFinal     = Join-Path $Root 'models\Pretrain\pretrain_final.pth'
$ChatBest    = Join-Path $Root 'models\Chat\model_best.pth'

Invoke-Stage -Name 'pretrain' `
             -Description 'Pretrain model' `
             -Command 'uv run python training/pretraining/pretrain.py' `
             -AlreadyDone (Test-Path $PtFinal) `
             -DoneReason "found $PtFinal"

# Stage 5 - Training (fine-tuning)
Invoke-Stage -Name 'train' `
             -Description 'Fine-tune chat model' `
             -Command 'uv run python training/train.py' `
             -AlreadyDone (Test-Path $ChatBest) `
             -DoneReason "found $ChatBest"

# Summary
$RunTimer.Stop()

Write-Log ""
Write-Log ("=" * 78) 'White'
Write-Log " Summary" 'White'
Write-Log ("=" * 78) 'White'

foreach ($s in $Summary) {
    $color = 'DarkGray'
    if ($s.Status -eq 'RAN') { $color = 'Green' }
    Write-Log ("  {0,-14} {1,-8} {2}" -f $s.Stage, $s.Status, $s.Duration) $color
}

Write-Log ""
Write-Log " total elapsed : $($RunTimer.Elapsed.ToString('hh\:mm\:ss'))" 'White'
Write-Log " final model   : $ChatBest" 'White'
Write-Log " log file      : $LogFile" 'White'
Write-Log ("=" * 78) 'White'

exit 0