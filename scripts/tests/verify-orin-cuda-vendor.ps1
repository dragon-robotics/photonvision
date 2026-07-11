$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$required = @(
    "native/orin-apriltag/UPSTREAM.md",
    "native/orin-apriltag/LICENSE-4143.txt",
    "native/orin-apriltag/VENDOR_MANIFEST.sha256",
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

function Get-RepoRelativePath {
    param([Parameter(Mandatory = $true)][string]$Path)

    return $Path.Substring($root.Length + 1).Replace("\", "/")
}

$vendorRoots = @(
    "native/orin-apriltag/upstream/frc971",
    "native/orin-apriltag/upstream/third_party"
)
$upstreamRoot = Join-Path $root "native/orin-apriltag/upstream"
$upstreamEntries = @(Get-ChildItem -LiteralPath $upstreamRoot -Recurse -Force)
$forbiddenEntries = @(
    $upstreamEntries | Where-Object {
        $relativePath = Get-RepoRelativePath -Path $_.FullName
        $segments = $relativePath.Split("/")
        $isGitMetadata = $segments -contains ".git"
        $isJniWrapper = $_.Name -eq "GpuDetectorJNI.cc"
        $isRootBuildFile = $relativePath -eq "native/orin-apriltag/upstream/CMakeLists.txt"
        $isBuildDirectory = $_.PSIsContainer -and (
            $_.Name -eq "build" -or
            $_.Name -eq "CMakeFiles" -or
            $_.Name -like "cmake-build-*"
        )
        $isBuildOutput = -not $_.PSIsContainer -and (
            $_.Name -in @(
                "CMakeCache.txt",
                "cmake_install.cmake",
                "install_manifest.txt",
                "build.ninja",
                ".ninja_deps",
                ".ninja_log"
            ) -or
            $_.Extension -in @(".o", ".obj", ".a", ".lib", ".so", ".dll", ".dylib", ".exe", ".class", ".jar", ".pyc")
        )

        $isGitMetadata -or $isJniWrapper -or $isRootBuildFile -or $isBuildDirectory -or $isBuildOutput
    } | ForEach-Object { Get-RepoRelativePath -Path $_.FullName }
)
if ($forbiddenEntries.Count -ne 0) {
    throw "Forbidden upstream JNI/build output or Git metadata: $($forbiddenEntries -join ', ')"
}

$unexpectedScopeFiles = @(
    $upstreamEntries | Where-Object { -not $_.PSIsContainer } | Where-Object {
        $relativePath = Get-RepoRelativePath -Path $_.FullName
        -not ($vendorRoots | Where-Object {
            $relativePath.StartsWith("$_/", [System.StringComparison]::Ordinal)
        })
    } | ForEach-Object { Get-RepoRelativePath -Path $_.FullName }
)
if ($unexpectedScopeFiles.Count -ne 0) {
    throw "Unexpected files outside approved vendor roots: $($unexpectedScopeFiles -join ', ')"
}

$vendoredFiles = @(
    $vendorRoots | ForEach-Object {
        Get-ChildItem -LiteralPath (Join-Path $root $_) -File -Recurse -Force
    }
)
$vendoredPaths = @($vendoredFiles | ForEach-Object { Get-RepoRelativePath -Path $_.FullName })
$vendoredPathSet = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::Ordinal)
$vendoredPaths | ForEach-Object { [void]$vendoredPathSet.Add($_) }

$manifestPath = Join-Path $root "native/orin-apriltag/VENDOR_MANIFEST.sha256"
$manifestEntries = New-Object 'System.Collections.Generic.Dictionary[string,string]' ([System.StringComparer]::Ordinal)
$invalidManifestLines = @()
$duplicateManifestPaths = @()
foreach ($line in Get-Content -LiteralPath $manifestPath) {
    if ($line -notmatch '^([0-9a-f]{64})  (.+)$') {
        $invalidManifestLines += $line
        continue
    }

    $hash = $Matches[1]
    $path = $Matches[2]
    $pathIsNormalized = -not $path.Contains("\") -and
        -not $path.StartsWith("/", [System.StringComparison]::Ordinal) -and
        -not $path.Contains("/../") -and
        -not $path.Contains("/./") -and
        -not $path.EndsWith("/..", [System.StringComparison]::Ordinal) -and
        -not $path.EndsWith("/.", [System.StringComparison]::Ordinal)
    if (-not $pathIsNormalized) {
        $invalidManifestLines += $line
        continue
    }

    if ($manifestEntries.ContainsKey($path)) {
        $duplicateManifestPaths += $path
        continue
    }

    $manifestEntries.Add($path, $hash)
}

if ($invalidManifestLines.Count -ne 0) {
    throw "Invalid vendor manifest lines: $($invalidManifestLines -join ', ')"
}
if ($duplicateManifestPaths.Count -ne 0) {
    throw "Duplicate vendor manifest paths: $($duplicateManifestPaths -join ', ')"
}

$manifestPathSet = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::Ordinal)
$manifestEntries.Keys | ForEach-Object { [void]$manifestPathSet.Add($_) }
$missingManifestPaths = @($vendoredPaths | Where-Object { -not $manifestPathSet.Contains($_) })
$extraManifestPaths = @($manifestEntries.Keys | Where-Object { -not $vendoredPathSet.Contains($_) })
if ($missingManifestPaths.Count -ne 0 -or $extraManifestPaths.Count -ne 0) {
    throw "Vendor manifest path set mismatch. Missing: $($missingManifestPaths -join ', '); Extra: $($extraManifestPaths -join ', ')"
}

$hashMismatches = @(
    $vendoredFiles | ForEach-Object {
        $relativePath = Get-RepoRelativePath -Path $_.FullName
        $actualHash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $manifestEntries[$relativePath]) {
            "$relativePath (expected $($manifestEntries[$relativePath]), actual $actualHash)"
        }
    }
)
if ($hashMismatches.Count -ne 0) {
    throw "Vendored CUDA file hash mismatch: $($hashMismatches -join ', ')"
}
