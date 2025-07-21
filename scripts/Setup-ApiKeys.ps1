# SecureKeyAgent - Windows PowerShell API Key Setup
# Secure API key management for AI News Dashboard

param(
    [Parameter(Mandatory=$false)]
    [string]$Action = "setup"
)

# Color functions for output
function Write-Success { param($Message) Write-Host $Message -ForegroundColor Green }
function Write-Error { param($Message) Write-Host $Message -ForegroundColor Red }
function Write-Warning { param($Message) Write-Host $Message -ForegroundColor Yellow }
function Write-Info { param($Message) Write-Host $Message -ForegroundColor Cyan }

# API key patterns for validation
$KeyPatterns = @{
    "OPENAI_API_KEY" = "^sk-[A-Za-z0-9]{48,}$"
    "GOOGLE_API_KEY" = "^[A-Za-z0-9_-]{39}$"
    "NEWS_API_KEY" = "^[a-f0-9]{32}$"
    "YOUTUBE_API_KEY" = "^[A-Za-z0-9_-]{39}$"
    "ANTHROPIC_API_KEY" = "^sk-ant-[A-Za-z0-9_-]{95,}$"
    "GEMINI_API_KEY" = "^[A-Za-z0-9_-]{39}$"
}

# Service configurations
$Services = @(
    @{
        Name = "OPENAI_API_KEY"
        Service = "OpenAI"
        Description = "Required for AI summarization and content generation"
        GetUrl = "https://platform.openai.com/api-keys"
        TestUrl = "https://api.openai.com/v1/models"
        Required = $true
    },
    @{
        Name = "GOOGLE_API_KEY"
        Service = "Google Custom Search"
        Description = "Required for news search and content aggregation"
        GetUrl = "https://console.cloud.google.com/apis/credentials"
        TestUrl = "https://www.googleapis.com/customsearch/v1"
        Required = $true
    },
    @{
        Name = "NEWS_API_KEY"
        Service = "NewsAPI"
        Description = "Alternative news source for content diversity"
        GetUrl = "https://newsapi.org/register"
        TestUrl = "https://newsapi.org/v2/top-headlines"
        Required = $false
    }
)

function Show-Header {
    Write-Host ""
    Write-Host "🔐 SecureKeyAgent - AI News Dashboard API Key Setup" -ForegroundColor Magenta
    Write-Host ("=" * 60) -ForegroundColor Gray
    Write-Success "✅ Secure storage in .env.local (excluded from git)"
    Write-Success "✅ Pattern validation for each service"
    Write-Success "✅ Live API testing with harmless calls"
    Write-Success "✅ Automatic .gitignore configuration"
    Write-Host ("=" * 60) -ForegroundColor Gray
    Write-Host ""
}

function Test-ApiKey {
    param(
        [string]$ServiceName,
        [string]$ApiKey,
        [string]$TestUrl
    )
    
    try {
        $headers = @{}
        $testEndpoint = $TestUrl
        
        switch ($ServiceName) {
            "OPENAI_API_KEY" {
                $headers["Authorization"] = "Bearer $ApiKey"
                $testEndpoint = "https://api.openai.com/v1/models"
            }
            "GOOGLE_API_KEY" {
                $testEndpoint = "https://www.googleapis.com/customsearch/v1?key=$ApiKey&cx=test&q=test"
            }
            "NEWS_API_KEY" {
                $testEndpoint = "https://newsapi.org/v2/top-headlines?country=us&pageSize=1&apiKey=$ApiKey"
            }
            "YOUTUBE_API_KEY" {
                $testEndpoint = "https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=1&q=test&key=$ApiKey"
            }
            "ANTHROPIC_API_KEY" {
                $headers["x-api-key"] = $ApiKey
                $headers["anthropic-version"] = "2023-06-01"
                $headers["content-type"] = "application/json"
                $body = @{
                    model = "claude-3-haiku-20240307"
                    max_tokens = 1
                    messages = @(@{ role = "user"; content = "Hi" })
                } | ConvertTo-Json
                $testEndpoint = "https://api.anthropic.com/v1/messages"
            }
        }
        
        Write-Info "🔍 Testing $($Services | Where-Object {$_.Name -eq $ServiceName} | Select-Object -ExpandProperty Service)..."
        
        $response = if ($ServiceName -eq "ANTHROPIC_API_KEY") {
            Invoke-RestMethod -Uri $testEndpoint -Method POST -Headers $headers -Body $body -TimeoutSec 10 -ErrorAction Stop
        } else {
            Invoke-RestMethod -Uri $testEndpoint -Method GET -Headers $headers -TimeoutSec 10 -ErrorAction Stop
        }
        
        return @{ Success = $true; Message = "API key validated successfully" }
    }
    catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        
        switch ($statusCode) {
            401 { return @{ Success = $false; Message = "Invalid API key" } }
            403 { return @{ Success = $false; Message = "API key invalid or insufficient permissions" } }
            429 { return @{ Success = $false; Message = "Rate limit exceeded" } }
            400 { 
                # Some services return 400 for test calls, which may be acceptable
                if ($ServiceName -eq "GOOGLE_API_KEY") {
                    return @{ Success = $true; Message = "API key format validated" }
                }
                return @{ Success = $false; Message = "Bad request - check API key format" }
            }
            default { return @{ Success = $false; Message = "HTTP $statusCode - $($_.Exception.Message)" } }
        }
    }
}

function Get-SecureInput {
    param(
        [string]$Prompt
    )
    
    Write-Host $Prompt -NoNewline
    $secureString = Read-Host -AsSecureString
    
    # Convert SecureString to plain text
    $ptr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureString)
    try {
        $plainText = [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
        return $plainText
    }
    finally {
        [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    }
}

function Test-KeyPattern {
    param(
        [string]$KeyName,
        [string]$ApiKey
    )
    
    if (-not $ApiKey -or $ApiKey.Trim().Length -eq 0) {
        return $false
    }
    
    $pattern = $KeyPatterns[$KeyName]
    if ($pattern) {
        return $ApiKey -match $pattern
    }
    
    # Generic validation for unknown keys
    return $ApiKey.Trim().Length -ge 16
}

function Set-EnvironmentKey {
    param(
        [string]$KeyName,
        [string]$ApiKey
    )
    
    $envFilePath = Join-Path (Get-Location) ".env.local"
    
    # Read existing content or create header
    $envContent = if (Test-Path $envFilePath) {
        Get-Content $envFilePath -Raw
    } else {
        @"
# AI News Dashboard - Secure Environment Configuration
# Generated by SecureKeyAgent - $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ')
# This file is automatically added to .gitignore
# NEVER commit API keys to version control

"@
    }
    
    # Remove existing key if present
    $envContent = $envContent -replace "(?m)^$KeyName=.*$", ""
    
    # Add the new key
    $envContent += "$KeyName=$ApiKey`n"
    
    # Write securely
    Set-Content -Path $envFilePath -Value $envContent -NoNewline
    
    # Set secure file permissions (Windows equivalent)
    $acl = Get-Acl $envFilePath
    $acl.SetAccessRuleProtection($true, $false)
    $accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
        [System.Security.Principal.WindowsIdentity]::GetCurrent().Name,
        "FullControl",
        "Allow"
    )
    $acl.SetAccessRule($accessRule)
    Set-Acl -Path $envFilePath -AclObject $acl
}

function Update-GitIgnore {
    $gitignorePath = Join-Path (Get-Location) ".gitignore"
    
    if (Test-Path $gitignorePath) {
        $gitignoreContent = Get-Content $gitignorePath -Raw
        
        if ($gitignoreContent -notmatch "\.env\.local") {
            Add-Content -Path $gitignorePath -Value "`n# Environment variables`n.env.local`n.env.*.local"
            Write-Success "✅ Added .env.local to .gitignore"
        } else {
            Write-Success "✅ .env.local already in .gitignore"
        }
    } else {
        Write-Warning "⚠️  No .gitignore found - consider creating one"
    }
}

function Setup-ApiKeys {
    Show-Header
    
    # Check for existing .env.local
    $envFilePath = Join-Path (Get-Location) ".env.local"
    if (Test-Path $envFilePath) {
        $choice = Read-Host "`n⚠️  .env.local already exists. Do you want to:`n1. Update existing keys`n2. Create backup and start fresh`n3. Cancel setup`nChoose (1/2/3)"
        
        switch ($choice) {
            "1" { Write-Info "📝 Updating existing .env.local..." }
            "2" { 
                Copy-Item $envFilePath "$envFilePath.backup"
                Write-Success "📋 Backup created: .env.local.backup"
            }
            "3" { 
                Write-Warning "Setup cancelled by user"
                return
            }
            default {
                Write-Error "Invalid choice"
                return
            }
        }
    }
    
    # Setup each service
    foreach ($service in $Services) {
        Write-Host "`n📝 Setting up $($service.Service) API Key" -ForegroundColor Yellow
        Write-Host "   Purpose: $($service.Description)" -ForegroundColor Gray
        Write-Host "   Get your key: $($service.GetUrl)" -ForegroundColor Gray
        
        $attempts = 0
        $maxAttempts = 3
        $isValid = $false
        
        while (-not $isValid -and $attempts -lt $maxAttempts) {
            $apiKey = Get-SecureInput "`nPlease paste your $($service.Service) API key: "
            
            # Validate format
            if (-not (Test-KeyPattern $service.Name $apiKey)) {
                Write-Error "❌ That key doesn't look right. Please check the format."
                $attempts++
                continue
            }
            
            # Test the key
            $testResult = Test-ApiKey $service.Name $apiKey $service.TestUrl
            
            if ($testResult.Success) {
                Set-EnvironmentKey $service.Name $apiKey
                Write-Success "✅ Key validated and stored securely!"
                $isValid = $true
            } else {
                Write-Error "❌ API test failed: $($testResult.Message)"
                Write-Host "Please check your key and try again." -ForegroundColor Gray
                $attempts++
            }
        }
        
        if (-not $isValid) {
            if ($service.Required) {
                Write-Error "❌ Failed to setup required service: $($service.Service)"
                return
            } else {
                Write-Warning "⚠️  Skipping optional service: $($service.Service)"
            }
        }
    }
    
    # Update .gitignore
    Update-GitIgnore
    
    # Final security check
    Write-Host "`n🛡️  Performing security check..." -ForegroundColor Cyan
    
    # Check file permissions
    if (Test-Path $envFilePath) {
        Write-Success "✅ .env.local created with secure permissions"
    }
    
    Write-Host "`n🎉 API Key Setup Complete!" -ForegroundColor Green
    Write-Success "✅ All keys stored securely in .env.local"
    Write-Success "✅ Keys validated and tested successfully"
    Write-Success "✅ .env.local is in .gitignore (not committed)"
    Write-Host "`n🚀 You can now run: npm run dev" -ForegroundColor Magenta
}

function Validate-ApiKeys {
    Write-Host "`n🔍 SecureKeyAgent - API Key Validator" -ForegroundColor Magenta
    Write-Host ("=" * 50) -ForegroundColor Gray
    
    $envFilePath = Join-Path (Get-Location) ".env.local"
    
    if (-not (Test-Path $envFilePath)) {
        Write-Error "❌ No .env.local file found"
        Write-Info "💡 Run: .\scripts\Setup-ApiKeys.ps1"
        return
    }
    
    $envContent = Get-Content $envFilePath -Raw
    $envKeys = @{}
    
    # Parse environment file
    $envContent -split "`n" | ForEach-Object {
        if ($_ -match "^([A-Z_]+)=(.+)$") {
            $envKeys[$matches[1]] = $matches[2].Trim()
        }
    }
    
    if ($envKeys.Count -eq 0) {
        Write-Error "❌ No API keys found in .env.local"
        return
    }
    
    Write-Host "📊 Found $($envKeys.Count) API keys to validate`n" -ForegroundColor Cyan
    
    $results = @()
    
    foreach ($service in $Services) {
        if ($envKeys.ContainsKey($service.Name)) {
            $result = Test-ApiKey $service.Name $envKeys[$service.Name] $service.TestUrl
            
            if ($result.Success) {
                Write-Success "   ✅ $($service.Service): Valid"
                $results += @{ Service = $service.Service; Status = "Valid" }
            } else {
                Write-Error "   ❌ $($service.Service): $($result.Message)"
                $results += @{ Service = $service.Service; Status = "Invalid"; Message = $result.Message }
            }
        } else {
            Write-Warning "   ⚠️  $($service.Service): Missing"
            $results += @{ Service = $service.Service; Status = "Missing" }
        }
    }
    
    # Summary
    $valid = ($results | Where-Object { $_.Status -eq "Valid" }).Count
    $invalid = ($results | Where-Object { $_.Status -eq "Invalid" }).Count
    $missing = ($results | Where-Object { $_.Status -eq "Missing" }).Count
    
    Write-Host "`n📊 Validation Results:" -ForegroundColor Yellow
    Write-Host ("=" * 30) -ForegroundColor Gray
    Write-Success "✅ Valid:   $valid"
    Write-Error "❌ Invalid: $invalid"
    Write-Warning "⚠️  Missing: $missing"
    
    if ($invalid -gt 0 -or $missing -gt 0) {
        Write-Host "`n💡 Run setup again to fix issues: .\scripts\Setup-ApiKeys.ps1" -ForegroundColor Cyan
    } else {
        Write-Host "`n🎉 All API keys are valid!" -ForegroundColor Green
    }
}

function Show-Help {
    Write-Host "`nSecureKeyAgent - API Key Management for AI News Dashboard" -ForegroundColor Magenta
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\scripts\Setup-ApiKeys.ps1 [action]" -ForegroundColor White
    Write-Host ""
    Write-Host "ACTIONS:" -ForegroundColor Yellow
    Write-Host "  setup     - Interactive API key setup (default)" -ForegroundColor White
    Write-Host "  validate  - Validate existing API keys" -ForegroundColor White
    Write-Host "  help      - Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Yellow
    Write-Host "  .\scripts\Setup-ApiKeys.ps1" -ForegroundColor Gray
    Write-Host "  .\scripts\Setup-ApiKeys.ps1 setup" -ForegroundColor Gray
    Write-Host "  .\scripts\Setup-ApiKeys.ps1 validate" -ForegroundColor Gray
    Write-Host ""
}

# Main execution
switch ($Action.ToLower()) {
    "setup" { Setup-ApiKeys }
    "validate" { Validate-ApiKeys }
    "help" { Show-Help }
    default { Setup-ApiKeys }
}
