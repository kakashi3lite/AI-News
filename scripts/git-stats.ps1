<#
.SYNOPSIS
    Git statistics script - shows repository statistics
#>

Write-Host "ðŸ“Š Git Repository Statistics" -ForegroundColor Blue
Write-Host "============================" -ForegroundColor Blue
Write-Host ""

# Repository info
Write-Host "ðŸ“ Repository Information:" -ForegroundColor Green
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
Write-Host "ðŸ“ˆ Commit Statistics:" -ForegroundColor Green
Write-Host "   Total commits: $(git rev-list --all --count)" -ForegroundColor White
Write-Host "   Commits this month: $(git rev-list --since='1 month ago' --count HEAD)" -ForegroundColor White
Write-Host "   Commits this week: $(git rev-list --since='1 week ago' --count HEAD)" -ForegroundColor White
Write-Host "   Commits today: $(git rev-list --since='1 day ago' --count HEAD)" -ForegroundColor White
Write-Host ""

# Author statistics
Write-Host "ðŸ‘¥ Top Contributors (last 3 months):" -ForegroundColor Green
git shortlog -sn --since="3 months ago" | Select-Object -First 10
Write-Host ""

# Branch information
Write-Host "ðŸŒ¿ Branch Information:" -ForegroundColor Green
$AllBranches = git branch -a
$LocalBranches = git branch
$RemoteBranches = git branch -r
Write-Host "   Total branches: $($AllBranches.Count)" -ForegroundColor White
Write-Host "   Local branches: $($LocalBranches.Count)" -ForegroundColor White
Write-Host "   Remote branches: $($RemoteBranches.Count)" -ForegroundColor White
Write-Host ""

# Recent activity
Write-Host "ðŸ•’ Recent Activity (last 10 commits):" -ForegroundColor Green
git log --oneline -10
Write-Host ""

# File statistics
Write-Host "ðŸ“„ File Statistics:" -ForegroundColor Green
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
Write-Host "ðŸ’¾ Repository Size:" -ForegroundColor Green
try {
    $GitDirSize = (Get-ChildItem -Path ".git" -Recurse -File | Measure-Object -Property Length -Sum).Sum
    $WorkingDirSize = (Get-ChildItem -Path "." -Recurse -File -Exclude ".git" | Measure-Object -Property Length -Sum).Sum
    Write-Host "   .git directory: $([math]::Round($GitDirSize / 1MB, 2)) MB" -ForegroundColor White
    Write-Host "   Working directory: $([math]::Round($WorkingDirSize / 1MB, 2)) MB" -ForegroundColor White
} catch {
    Write-Host "   Size calculation failed" -ForegroundColor Yellow
}
