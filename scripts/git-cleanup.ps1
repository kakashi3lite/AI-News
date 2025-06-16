<#
.SYNOPSIS
    Git cleanup script - removes merged branches and cleans up repository
#>

Write-Host "ðŸ§¹ Cleaning up Git repository..." -ForegroundColor Green

# Fetch latest changes
Write-Host "ðŸ“¡ Fetching latest changes..." -ForegroundColor Blue
git fetch --all --prune

# Clean up merged branches
Write-Host "ðŸŒ¿ Removing merged branches..." -ForegroundColor Blue
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
Write-Host "ðŸ”— Cleaning up remote tracking branches..." -ForegroundColor Blue
git remote prune origin

# Clean up stale references
Write-Host "ðŸ—‘ï¸  Cleaning up stale references..." -ForegroundColor Blue
git gc --prune=now

# Show current status
Write-Host "ðŸ“Š Repository status:" -ForegroundColor Blue
git status -s
Write-Host "ðŸŒ¿ Remaining branches:" -ForegroundColor Blue
git branch -a

Write-Host "[SUCCESS] Cleanup complete!" -ForegroundColor Green
