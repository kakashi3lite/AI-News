<#
.SYNOPSIS
    Git Setup Script for AI News Dashboard (PowerShell)
    Configures Git hooks, aliases, and project-specific settings

.DESCRIPTION
    This script automates the setup of Git configuration for the AI News Dashboard project.
    It configures Git hooks, useful aliases, project-specific settings, and development tools.

.PARAMETER Force
    Force overwrite existing configurations

.PARAMETER Minimal
    Install minimal configuration only

.PARAMETER Help
    Show help information

.EXAMPLE
    .\scripts\setup-git.ps1
    Run with default settings

.EXAMPLE
    .\scripts\setup-git.ps1 -Force
    Force overwrite existing configurations

.EXAMPLE
    .\scripts\setup-git.ps1 -Minimal
    Install minimal configuration only
#>

param(
    [switch]$Force,
    [switch]$Minimal,
    [switch]$Help
)

# Configuration
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$GitHooksDir = Join-Path $ProjectRoot ".githooks"
$ScriptsDir = Join-Path $ProjectRoot "scripts"

# Colors for output (Windows PowerShell compatible)
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    Purple = "Magenta"
    Cyan = "Cyan"
    White = "White"
}

# Helper functions
function Write-Header {
    param([string]$Message)
    Write-Host "`n=== $Message ===" -ForegroundColor $Colors.Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor $Colors.Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor $Colors.Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor $Colors.Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor $Colors.Cyan
}

function Write-Step {
    param([string]$Message)
    Write-Host "→ $Message" -ForegroundColor $Colors.Purple
}

# Show help if requested
if ($Help) {
    Write-Host "Git Setup Script for AI News Dashboard" -ForegroundColor $Colors.Blue
    Write-Host ""
    Write-Host "Usage: .\scripts\setup-git.ps1 [options]" -ForegroundColor $Colors.White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor $Colors.White
    Write-Host "  -Force    Force overwrite existing configurations" -ForegroundColor $Colors.White
    Write-Host "  -Minimal  Install minimal configuration only" -ForegroundColor $Colors.White
    Write-Host "  -Help     Show this help message" -ForegroundColor $Colors.White
    Write-Host ""
    Write-Host "This script will:" -ForegroundColor $Colors.White
    Write-Host "  - Configure Git hooks for code quality" -ForegroundColor $Colors.White
    Write-Host "  - Set up useful Git aliases" -ForegroundColor $Colors.White
    Write-Host "  - Configure project-specific Git settings" -ForegroundColor $Colors.White
    Write-Host "  - Install commit message templates" -ForegroundColor $Colors.White
    Write-Host "  - Set up branch protection workflows" -ForegroundColor $Colors.White
    exit 0
}

# Check if we're in a Git repository
try {
    git rev-parse --git-dir | Out-Null
} catch {
    Write-Error "Not in a Git repository. Please run this script from the project root."
    exit 1
}

Write-Header "AI News Dashboard Git Setup"
Write-Info "Project root: $ProjectRoot"
Write-Info "Git hooks directory: $GitHooksDir"
Write-Info "Force mode: $Force"
Write-Info "Minimal mode: $Minimal"

# 1. Create .githooks directory if it doesn't exist
Write-Header "Setting up Git hooks directory"
if (-not (Test-Path $GitHooksDir)) {
    New-Item -ItemType Directory -Path $GitHooksDir -Force | Out-Null
    Write-Success "Created .githooks directory"
} else {
    Write-Info ".githooks directory already exists"
}

# 2. Configure Git to use custom hooks directory
Write-Step "Configuring Git hooks path"
try {
    git config core.hooksPath $GitHooksDir
    Write-Success "Git hooks path configured to $GitHooksDir"
} catch {
    Write-Error "Failed to configure Git hooks path"
    exit 1
}

# 3. Make hooks executable (Windows doesn't need chmod, but we can check if they exist)
Write-Step "Checking hooks"
Get-ChildItem -Path $GitHooksDir -File | ForEach-Object {
    Write-Success "Found hook: $($_.Name)"
}

# 4. Set up commit message template
Write-Header "Configuring commit message template"
$GitMessagePath = Join-Path $ProjectRoot ".gitmessage"
if (Test-Path $GitMessagePath) {
    try {
        git config commit.template $GitMessagePath
        Write-Success "Commit message template configured"
    } catch {
        Write-Warning "Failed to configure commit message template"
    }
} else {
    Write-Warning "Commit message template file not found"
}

# 5. Configure useful Git aliases
Write-Header "Setting up Git aliases"

# Define aliases
$Aliases = @{
    "st" = "status -s"
    "co" = "checkout"
    "br" = "branch"
    "ci" = "commit"
    "ca" = "commit -a"
    "cm" = "commit -m"
    "cam" = "commit -am"
    "unstage" = "reset HEAD --"
    "last" = "log -1 HEAD"
    "visual" = "!gitk"
    "graph" = "log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
    "lg" = "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
    "contributors" = "shortlog --summary --numbered"
    "amend" = "commit --amend --no-edit"
    "fixup" = "commit --fixup"
    "squash" = "commit --squash"
    "wip" = "commit -am 'WIP: work in progress'"
    "unwip" = "reset HEAD~1"
    "aliases" = "config --get-regexp alias"
    "branches" = "branch -a"
    "remotes" = "remote -v"
    "tags" = "tag -l"
    "stashes" = "stash list"
    "conflicts" = "diff --name-only --diff-filter=U"
    "resolve" = "add -u"
    "abort" = "merge --abort"
    "continue" = "rebase --continue"
    "skip" = "rebase --skip"
    "edit" = "rebase --edit-todo"
    "root" = "rev-parse --show-toplevel"
    "exec" = "!exec "
    "feature" = "checkout -b feature/"
    "bugfix" = "checkout -b bugfix/"
    "hotfix" = "checkout -b hotfix/"
    "release" = "checkout -b release/"
    "develop" = "checkout develop"
    "main" = "checkout main"
    "master" = "checkout master"
    "sync" = "!git fetch origin && git rebase origin/$(git branch --show-current)"
    "update" = "!git fetch origin && git merge origin/$(git branch --show-current)"
    "clean-branches" = "!git branch --merged | Where-Object { $_ -notmatch '\*|main|master|develop' } | ForEach-Object { git branch -d $_.Trim() }"
    "recent" = "branch --sort=-committerdate"
    "find" = "!git ls-files | Select-String -Pattern"
    "grep" = "grep -Ii"
    "la" = "!git config -l | Select-String alias | ForEach-Object { $_.Line.Substring(6) }"
    "save" = "!git add -A && git commit -m 'SAVEPOINT'"
    "undo" = "reset HEAD~1 --mixed"
    "res" = "!git reset --hard"
    "reshard" = "reset --hard"
    "type" = "cat-file -t"
    "dump" = "cat-file -p"
    "look" = "log --oneline --decorate"
    "overview" = "log --all --since='2 weeks' --oneline --no-merges"
    "recap" = "!git log --all --oneline --no-merges --author=$(git config user.email)"
    "today" = "!git log --since=00:00:00 --all --no-merges --oneline --author=$(git config user.email)"
    "changelog" = "log --oneline --no-merges"
}

# Install aliases
$AliasCount = 0
foreach ($AliasName in $Aliases.Keys) {
    $AliasCommand = $Aliases[$AliasName]
    
    # Check if alias already exists
    try {
        $ExistingAlias = git config --get alias.$AliasName 2>$null
        if ($ExistingAlias -and -not $Force) {
            Write-Info "Alias '$AliasName' already exists (use -Force to overwrite)"
        } else {
            git config alias.$AliasName $AliasCommand
            if ($ExistingAlias) {
                Write-Success "Updated alias: $AliasName"
            } else {
                Write-Success "Added alias: $AliasName"
            }
            $AliasCount++
        }
    } catch {
        git config alias.$AliasName $AliasCommand
        Write-Success "Added alias: $AliasName"
        $AliasCount++
    }
}

Write-Info "Configured $AliasCount aliases"

# 6. Configure Git settings for the project
Write-Header "Configuring Git settings"

# Core settings
Write-Step "Setting core configurations"
git config core.autocrlf false
git config core.safecrlf true
git config core.filemode false
git config core.ignorecase false
git config core.precomposeunicode true
git config core.quotepath false
Write-Success "Core settings configured"

# Push settings
Write-Step "Setting push configurations"
git config push.default simple
git config push.followTags true
Write-Success "Push settings configured"

# Pull settings
Write-Step "Setting pull configurations"
git config pull.rebase true
git config rebase.autoStash true
Write-Success "Pull settings configured"

# Merge settings
Write-Step "Setting merge configurations"
git config merge.conflictstyle diff3
Write-Success "Merge settings configured"

# Branch settings
Write-Step "Setting branch configurations"
git config branch.autosetupmerge always
git config branch.autosetuprebase always
Write-Success "Branch settings configured"

# Color settings
if (-not $Minimal) {
    Write-Step "Setting color configurations"
    git config color.ui auto
    git config color.branch auto
    git config color.diff auto
    git config color.status auto
    git config color.grep auto
    git config color.interactive auto
    Write-Success "Color settings configured"
}

# 7. Set up additional configurations for development
if (-not $Minimal) {
    Write-Header "Setting up development configurations"
    
    # Configure diff and merge tools
    Write-Step "Configuring diff and merge tools"
    
    # Check for VS Code
    if (Get-Command code -ErrorAction SilentlyContinue) {
        git config diff.tool vscode
        git config difftool.vscode.cmd 'code --wait --diff $LOCAL $REMOTE'
        git config merge.tool vscode
        git config mergetool.vscode.cmd 'code --wait $MERGED'
        Write-Success "VS Code configured as diff/merge tool"
    }
    
    # Configure rerere (reuse recorded resolution)
    git config rerere.enabled true
    Write-Success "Rerere enabled for conflict resolution"
    
    # Configure help settings
    git config help.autocorrect 1
    Write-Success "Help autocorrect enabled"
    
    # Configure log settings
    git config log.date relative
    Write-Success "Log date format set to relative"
}

# 8. Create useful Git scripts
Write-Header "Creating utility scripts"

# Create git-cleanup.ps1 script
$GitCleanupScript = @'
<#
.SYNOPSIS
    Git cleanup script - removes merged branches and cleans up repository
#>

Write-Host "🧹 Cleaning up Git repository..." -ForegroundColor Green

# Fetch latest changes
Write-Host "📡 Fetching latest changes..." -ForegroundColor Blue
git fetch --all --prune

# Clean up merged branches
Write-Host "🌿 Removing merged branches..." -ForegroundColor Blue
try {
    $MergedBranches = git branch --merged | Where-Object { $_ -notmatch "\*|main|master|develop" }
    if ($MergedBranches) {
        $MergedBranches | ForEach-Object {
            $BranchName = $_.Trim()
            if ($BranchName) {
                git branch -d $BranchName
                Write-Host "  Deleted: $BranchName" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "  No merged branches to delete" -ForegroundColor Green
    }
} catch {
    Write-Host "  Error cleaning up branches: $($_.Exception.Message)" -ForegroundColor Red
}

# Clean up remote tracking branches
Write-Host "🔗 Cleaning up remote tracking branches..." -ForegroundColor Blue
git remote prune origin

# Clean up stale references
Write-Host "🗑️  Cleaning up stale references..." -ForegroundColor Blue
git gc --prune=now

# Show current status
Write-Host "📊 Repository status:" -ForegroundColor Blue
git status -s
Write-Host "🌿 Remaining branches:" -ForegroundColor Blue
git branch -a

Write-Host "✅ Cleanup complete!" -ForegroundColor Green
'@

$GitCleanupPath = Join-Path $ScriptsDir "git-cleanup.ps1"
Set-Content -Path $GitCleanupPath -Value $GitCleanupScript -Encoding UTF8
Write-Success "Created git-cleanup.ps1 script"

# Create git-stats.ps1 script
$GitStatsScript = @'
<#
.SYNOPSIS
    Git statistics script - shows repository statistics
#>

Write-Host "📊 Git Repository Statistics" -ForegroundColor Blue
Write-Host "============================" -ForegroundColor Blue
Write-Host ""

# Repository info
Write-Host "📁 Repository Information:" -ForegroundColor Green
Write-Host "   Root: $(git rev-parse --show-toplevel)" -ForegroundColor White
Write-Host "   Branch: $(git branch --show-current)" -ForegroundColor White
try {
    $RemoteUrl = git remote get-url origin 2>$null
    Write-Host "   Remote: $RemoteUrl" -ForegroundColor White
} catch {
    Write-Host "   Remote: No remote configured" -ForegroundColor Yellow
}
Write-Host ""

# Commit statistics
Write-Host "📈 Commit Statistics:" -ForegroundColor Green
Write-Host "   Total commits: $(git rev-list --all --count)" -ForegroundColor White
Write-Host "   Commits this month: $(git rev-list --since='1 month ago' --count HEAD)" -ForegroundColor White
Write-Host "   Commits this week: $(git rev-list --since='1 week ago' --count HEAD)" -ForegroundColor White
Write-Host "   Commits today: $(git rev-list --since='1 day ago' --count HEAD)" -ForegroundColor White
Write-Host ""

# Author statistics
Write-Host "👥 Top Contributors (last 3 months):" -ForegroundColor Green
git shortlog -sn --since="3 months ago" | Select-Object -First 10
Write-Host ""

# Branch information
Write-Host "🌿 Branch Information:" -ForegroundColor Green
$AllBranches = git branch -a
$LocalBranches = git branch
$RemoteBranches = git branch -r
Write-Host "   Total branches: $($AllBranches.Count)" -ForegroundColor White
Write-Host "   Local branches: $($LocalBranches.Count)" -ForegroundColor White
Write-Host "   Remote branches: $($RemoteBranches.Count)" -ForegroundColor White
Write-Host ""

# Recent activity
Write-Host "🕒 Recent Activity (last 10 commits):" -ForegroundColor Green
git log --oneline -10
Write-Host ""

# File statistics
Write-Host "📄 File Statistics:" -ForegroundColor Green
$FileCount = (git ls-files).Count
Write-Host "   Total files: $FileCount" -ForegroundColor White
Write-Host "   Largest files:" -ForegroundColor White
$LargestFiles = git ls-tree -r -l HEAD | Sort-Object { [int]($_ -split "\s+")[3] } -Descending | Select-Object -First 5
$LargestFiles | ForEach-Object {
    $Parts = $_ -split "\s+"
    $Size = $Parts[3]
    $Name = $Parts[4]
    Write-Host "     $Size bytes - $Name" -ForegroundColor Cyan
}
Write-Host ""

# Repository size
Write-Host "💾 Repository Size:" -ForegroundColor Green
try {
    $GitDirSize = (Get-ChildItem -Path ".git" -Recurse -File | Measure-Object -Property Length -Sum).Sum
    $WorkingDirSize = (Get-ChildItem -Path "." -Recurse -File -Exclude ".git" | Measure-Object -Property Length -Sum).Sum
    Write-Host "   .git directory: $([math]::Round($GitDirSize / 1MB, 2)) MB" -ForegroundColor White
    Write-Host "   Working directory: $([math]::Round($WorkingDirSize / 1MB, 2)) MB" -ForegroundColor White
} catch {
    Write-Host "   Size calculation failed" -ForegroundColor Yellow
}
'@

$GitStatsPath = Join-Path $ScriptsDir "git-stats.ps1"
Set-Content -Path $GitStatsPath -Value $GitStatsScript -Encoding UTF8
Write-Success "Created git-stats.ps1 script"

# 9. Test Git hooks
Write-Header "Testing Git hooks"

$HookFiles = @("pre-commit", "commit-msg", "pre-push")
foreach ($HookFile in $HookFiles) {
    $HookPath = Join-Path $GitHooksDir $HookFile
    if (Test-Path $HookPath) {
        Write-Step "Testing $HookFile hook"
        Write-Success "$HookFile hook exists"
    }
}

# 10. Create .gitattributes if it doesn't exist
Write-Header "Setting up .gitattributes"

$GitAttributesPath = Join-Path $ProjectRoot ".gitattributes"
if (-not (Test-Path $GitAttributesPath) -or $Force) {
    $GitAttributesContent = @'
# Auto detect text files and perform LF normalization
* text=auto

# Explicitly declare text files you want to always be normalized and converted
# to native line endings on checkout.
*.c text
*.h text
*.js text
*.ts text
*.jsx text
*.tsx text
*.json text
*.md text
*.txt text
*.yml text
*.yaml text
*.xml text
*.html text
*.css text
*.scss text
*.sass text
*.less text
*.sql text
*.sh text eol=lf
*.bat text eol=crlf
*.ps1 text eol=crlf

# Declare files that will always have CRLF line endings on checkout.
*.sln text eol=crlf

# Denote all files that are truly binary and should not be modified.
*.png binary
*.jpg binary
*.jpeg binary
*.gif binary
*.ico binary
*.mov binary
*.mp4 binary
*.mp3 binary
*.flv binary
*.fla binary
*.swf binary
*.gz binary
*.zip binary
*.7z binary
*.ttf binary
*.eot binary
*.woff binary
*.woff2 binary
*.pyc binary
*.pdf binary
*.ez binary
*.bz2 binary
*.swp binary
*.jar binary
*.class binary
*.so binary
*.dll binary
*.exe binary

# Language-specific settings
*.js linguist-language=JavaScript
*.ts linguist-language=TypeScript
*.jsx linguist-language=JavaScript
*.tsx linguist-language=TypeScript

# Documentation
*.md linguist-documentation
*.txt linguist-documentation
README* linguist-documentation
CHANGELOG* linguist-documentation
CONTRIBUTING* linguist-documentation
LICENSE* linguist-documentation

# Generated files
*.min.js linguist-generated
*.min.css linguist-generated
*.map linguist-generated
dist/* linguist-generated
build/* linguist-generated
out/* linguist-generated
.next/* linguist-generated
coverage/* linguist-generated
node_modules/* linguist-vendored
vendor/* linguist-vendored
'@
    Set-Content -Path $GitAttributesPath -Value $GitAttributesContent -Encoding UTF8
    Write-Success "Created .gitattributes file"
} else {
    Write-Info ".gitattributes already exists (use -Force to overwrite)"
}

# 11. Final summary and next steps
Write-Header "Setup Complete!"

Write-Success "Git configuration has been successfully set up for AI News Dashboard"
Write-Host ""
Write-Info "What was configured:"
Write-Host "  ✓ Git hooks directory and permissions" -ForegroundColor Green
Write-Host "  ✓ Commit message template" -ForegroundColor Green
Write-Host "  ✓ Useful Git aliases ($($Aliases.Count) aliases)" -ForegroundColor Green
Write-Host "  ✓ Project-specific Git settings" -ForegroundColor Green
Write-Host "  ✓ Development tools configuration" -ForegroundColor Green
Write-Host "  ✓ Utility scripts for cleanup and statistics" -ForegroundColor Green
Write-Host "  ✓ .gitattributes for proper file handling" -ForegroundColor Green
Write-Host ""
Write-Info "Available commands:"
Write-Host "  git aliases                    - List all configured aliases" -ForegroundColor Cyan
Write-Host "  .\scripts\git-cleanup.ps1      - Clean up merged branches" -ForegroundColor Cyan
Write-Host "  .\scripts\git-stats.ps1        - Show repository statistics" -ForegroundColor Cyan
Write-Host ""
Write-Info "Useful aliases to try:"
Write-Host "  git st              - Short status" -ForegroundColor Yellow
Write-Host "  git lg              - Pretty log graph" -ForegroundColor Yellow
Write-Host "  git sync            - Sync with remote branch" -ForegroundColor Yellow
Write-Host "  git clean-branches  - Remove merged branches" -ForegroundColor Yellow
Write-Host "  git wip             - Quick work-in-progress commit" -ForegroundColor Yellow
Write-Host "  git recent          - Show recent branches" -ForegroundColor Yellow
Write-Host ""
Write-Info "Next steps:"
Write-Host "  1. Test the setup with a small commit" -ForegroundColor White
Write-Host "  2. Review the Git hooks in .githooks/" -ForegroundColor White
Write-Host "  3. Customize aliases as needed" -ForegroundColor White
Write-Host "  4. Share this setup with your team" -ForegroundColor White
Write-Host ""
Write-Success "Happy coding! 🚀"