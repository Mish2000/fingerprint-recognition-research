param(
    [string]$Repo = 'C:\fingerprint-recognition-research',
    [string]$Out = ''
)

$Repo = (Resolve-Path -LiteralPath $Repo).Path.TrimEnd('\', '/')
if ([string]::IsNullOrWhiteSpace($Out)) {
    $Out = Join-Path $Repo 'ai_repo_structure.txt'
}

$ExcludeDirs = @(
    '.git',
    '.idea',
    '.vscode',
    '__pycache__',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
    'node_modules',
    'venv',
    '.venv',
    'env',
    '.env',
    'dist',
    'build',
    'target',
    'out',
    'template_cache',
    'sourceafis_template_cache',
    'candidate_score_cache',
    'visual_audit_sheets',
    '.vite',
    'htmlcov',
    '.ipynb_checkpoints',
    'tmp',
    'captures',
    'tmp_captures'
)

$ExcludeFiles = @(
    'ai_repo_structure.txt',
    'project_tree.txt'
)

$ExcludeDirSet = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
foreach ($dir in $ExcludeDirs) {
    [void]$ExcludeDirSet.Add($dir)
}

$ExcludeFileSet = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
foreach ($file in $ExcludeFiles) {
    [void]$ExcludeFileSet.Add($file)
}

function Get-RepoIndexItem {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    foreach ($item in Get-ChildItem -LiteralPath $Path -Force -ErrorAction SilentlyContinue) {
        if ($item.PSIsContainer) {
            if ($ExcludeDirSet.Contains($item.Name)) {
                continue
            }

            $item
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                continue
            }

            Get-RepoIndexItem -Path $item.FullName
            continue
        }

        if ($ExcludeFileSet.Contains($item.Name)) {
            continue
        }

        $item
    }
}

function Get-RelativePathForIndex {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.FileSystemInfo]$Item
    )

    return $Item.FullName.Substring($Repo.Length).TrimStart([char[]]@('\', '/'))
}

$Items = @(Get-RepoIndexItem -Path $Repo | Sort-Object FullName)

$Utf8NoBom = New-Object System.Text.UTF8Encoding $false
$Writer = [System.IO.StreamWriter]::new($Out, $false, $Utf8NoBom)

try {
    $Writer.WriteLine('# AI Repository Structure Index')
    $Writer.WriteLine('')
    $Writer.WriteLine("Repository root: $Repo")
    $Writer.WriteLine("Generated at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
    $Writer.WriteLine('')
    $Writer.WriteLine('Purpose:')
    $Writer.WriteLine('This file is designed for an AI agent to quickly understand the repository layout and retrieve exact file/folder paths.')
    $Writer.WriteLine('Reparse point directories are listed but not traversed.')
    $Writer.WriteLine('')
    $Writer.WriteLine('Excluded noisy directories:')
    $Writer.WriteLine(($ExcludeDirs -join ', '))
    $Writer.WriteLine('')
    $Writer.WriteLine('Excluded generated files:')
    $Writer.WriteLine(($ExcludeFiles -join ', '))
    $Writer.WriteLine('')
    $Writer.WriteLine('Legend:')
    $Writer.WriteLine('DIR  = directory')
    $Writer.WriteLine('FILE = file')
    $Writer.WriteLine('')
    $Writer.WriteLine('============================================================')
    $Writer.WriteLine('SECTION 1 - HUMAN READABLE TREE')
    $Writer.WriteLine('============================================================')
    $Writer.WriteLine('')

    foreach ($item in $Items) {
        $rel = Get-RelativePathForIndex -Item $item
        if ([string]::IsNullOrWhiteSpace($rel)) {
            continue
        }

        $depth = ($rel -split '[\\/]+').Count - 1
        $indent = '  ' * $depth

        if ($item.PSIsContainer) {
            $Writer.WriteLine("$indent[D] $($item.Name)/")
        } else {
            $Writer.WriteLine("$indent[F] $($item.Name)")
        }
    }

    $Writer.WriteLine('')
    $Writer.WriteLine('============================================================')
    $Writer.WriteLine('SECTION 2 - MACHINE READABLE PATH INDEX')
    $Writer.WriteLine('============================================================')
    $Writer.WriteLine('')
    $Writer.WriteLine("TYPE`tDEPTH`tEXTENSION`tSIZE_BYTES`tLAST_WRITE_TIME`tRELATIVE_PATH")

    foreach ($item in $Items) {
        $rel = Get-RelativePathForIndex -Item $item
        if ([string]::IsNullOrWhiteSpace($rel)) {
            continue
        }

        $type = if ($item.PSIsContainer) { 'DIR' } else { 'FILE' }
        $depth = ($rel -split '[\\/]+').Count - 1
        $ext = if ($item.PSIsContainer) { '' } else { $item.Extension }
        $size = if ($item.PSIsContainer) { '' } else { $item.Length }
        $lastWrite = $item.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')

        $Writer.WriteLine("$type`t$depth`t$ext`t$size`t$lastWrite`t$rel")
    }
} finally {
    $Writer.Dispose()
}

Write-Host 'Created AI repo structure file:'
Write-Host $Out
Write-Host "Indexed items: $($Items.Count)"
