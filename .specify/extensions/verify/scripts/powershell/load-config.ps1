# load-config.ps1 — Load and validate the verify extension configuration.
#
# Reads report.max_findings from the YAML config file,
# normalises YAML null sentinels, applies an optional environment
# variable override (SPECKIT_VERIFY_MAX_FINDINGS), and validates
# that a value is present before exporting it.
#
# Usage:  load-config.ps1
#
# Exit codes:
#   0 — configuration loaded successfully
#   1 — config file missing, required value not set, or invalid value

$configFile = ".specify/extensions/verify/verify-config.yml"
$extensionFile = ".specify/extensions/verify/extension.yml"
$usingDefaults = $false

if (-not (Test-Path $configFile)) {
    if (Test-Path $extensionFile) {
        $usingDefaults = $true
    } else {
        Write-Error "❌ Error: Configuration not found at $configFile"
        Write-Error "Run 'specify extension add verify' to install and configure"
        exit 1
    }
}

# Read configuration values

# Extract a YAML value for a key from a file using only built-in tools.
function Get-YamlValue {
    param([string]$Key, [string]$File)
    $lines = Get-Content $File -ErrorAction SilentlyContinue
    if (-not $lines) { return '' }
    $match = $lines | Select-String -Pattern "^\s*${Key}:" | Select-Object -Last 1
    if (-not $match) { return '' }
    $raw = $match.Line -replace '^[^:]*:', ''
    $raw = $raw.Trim()
    $hasDoubleQuotes = $raw.Length -ge 2 -and $raw.StartsWith('"') -and $raw.EndsWith('"')
    $hasSingleQuotes = $raw.Length -ge 2 -and $raw.StartsWith("'") -and $raw.EndsWith("'")

    if ($hasDoubleQuotes -or $hasSingleQuotes) {
        $quoteChar = $raw.Substring(0, 1)
        $raw = $raw.Substring(1, $raw.Length - 2)
        if ($quoteChar -eq '"') {
            $raw = $raw.Replace('""', '"')
        } else {
            $raw = $raw.Replace("''", "'")
        }
    } else {
        $raw = ($raw -split '#', 2)[0]
    }

    return $raw.Trim()
}

if ($usingDefaults) {
    $maxFindings = Get-YamlValue -Key 'max_findings' -File $extensionFile
} else {
    $maxFindings = Get-YamlValue -Key 'max_findings' -File $configFile
}

# Treat YAML null sentinels as empty
if ($maxFindings -eq 'null' -or $maxFindings -eq '~') {
    $maxFindings = ''
}

# Apply environment variable overrides

if ($null -ne $env:SPECKIT_VERIFY_MAX_FINDINGS -and $env:SPECKIT_VERIFY_MAX_FINDINGS -ne '') {
    $maxFindings = $env:SPECKIT_VERIFY_MAX_FINDINGS
}

# Validate configuration

if (-not $maxFindings) {
    Write-Error "❌ Error: Configuration value not set"
    Write-Error "Edit $configFile and set 'report.max_findings'"
    exit 1
}

if ($maxFindings -notmatch '^\d+$') {
    Write-Error "❌ Error: 'report.max_findings' must be a non-negative integer (>= 0), got '$maxFindings'"
    Write-Error "Edit $configFile and set 'report.max_findings' to 0 (unlimited) or a positive integer (e.g. 50)"
    exit 1
}

if ($usingDefaults) {
    Write-Warning "No config file found; using defaults from extension.yml"
}

Write-Output "📋 Configuration loaded: max_findings=$maxFindings"
