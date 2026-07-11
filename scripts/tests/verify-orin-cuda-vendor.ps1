$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "../..")
$required = @(
    "native/orin-apriltag/UPSTREAM.md",
    "native/orin-apriltag/LICENSE-4143.txt",
    "native/orin-apriltag/upstream/frc971/orin/971apriltag.h",
    "native/orin-apriltag/upstream/frc971/orin/971apriltag.cu",
    "native/orin-apriltag/upstream/third_party/apriltag/LICENSE.md",
    "native/orin-apriltag/upstream/third_party/apriltag/apriltag.h"
)

$missing = $required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $root $_)) }
if ($missing.Count -ne 0) {
    throw "Missing vendored CUDA files: $($missing -join ', ')"
}

$provenance = Get-Content -LiteralPath (Join-Path $root "native/orin-apriltag/UPSTREAM.md") -Raw
if (-not $provenance.Contains("ef9fc1ec7e43116849e71fef1ab335ba630274a7")) {
    throw "UPSTREAM.md does not pin the approved GpuDetectorJNI revision"
}
