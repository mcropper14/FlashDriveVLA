param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
)

$ErrorActionPreference = "Stop"

$dirLinks = @(
  @{ Link = "msgq"; Target = "msgq_repo\msgq" },
  @{ Link = "opendbc"; Target = "opendbc_repo\opendbc" },
  @{ Link = "rednose"; Target = "rednose_repo\rednose" },
  @{ Link = "tinygrad"; Target = "tinygrad_repo\tinygrad" },
  @{ Link = "teleoprtc"; Target = "teleoprtc_repo\teleoprtc" },
  @{ Link = "openpilot\common"; Target = "common" },
  @{ Link = "openpilot\selfdrive"; Target = "selfdrive" },
  @{ Link = "openpilot\system"; Target = "system" },
  @{ Link = "openpilot\tools"; Target = "tools" }
)

foreach ($entry in $dirLinks) {
  $link = Join-Path $RepoRoot $entry.Link
  $target = Join-Path $RepoRoot $entry.Target
  $resolvedTarget = (Resolve-Path -LiteralPath $target).Path
  if (-not $resolvedTarget.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to link outside repo: $resolvedTarget"
  }
  if (Test-Path -LiteralPath $link) {
    Remove-Item -LiteralPath $link -Force
  }
  New-Item -ItemType Junction -Path $link -Target $resolvedTarget | Out-Null
}

$carLink = Join-Path $RepoRoot "cereal\car.capnp"
$carTarget = Join-Path $RepoRoot "opendbc_repo\opendbc\car\car.capnp"
$resolvedCarTarget = (Resolve-Path -LiteralPath $carTarget).Path
if (-not $resolvedCarTarget.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
  throw "Refusing to link outside repo: $resolvedCarTarget"
}
if (Test-Path -LiteralPath $carLink) {
  Remove-Item -LiteralPath $carLink -Force
}

try {
  New-Item -ItemType SymbolicLink -Path $carLink -Target $resolvedCarTarget -ErrorAction Stop | Out-Null
} catch {
  Copy-Item -LiteralPath $resolvedCarTarget -Destination $carLink
}

Write-Host "Windows openpilot symlink compatibility links prepared in $RepoRoot"
