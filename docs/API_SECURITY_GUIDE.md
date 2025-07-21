# 🔐 API Security & Key Management Guide

## 🎯 SecureKeyAgent Overview

The **SecureKeyAgent** is an enterprise-grade API key management system designed for the AI News Dashboard. It provides secure storage, validation, testing, and management of API keys with comprehensive security features.

## 🚀 Quick Start

### **Method 1: Interactive Setup (Recommended)**
```bash
# For all platforms
npm run setup:keys

# Windows-specific (PowerShell)
npm run setup:keys:windows
```

### **Method 2: Manual Script Execution**
```bash
# Node.js version (cross-platform)
node scripts/secure-key-setup.js

# PowerShell version (Windows)
.\scripts\Setup-ApiKeys.ps1
```

## 🔧 Available Commands

### **Setup Commands**
- `npm run setup:keys` - Interactive API key setup with validation
- `npm run setup:keys:windows` - Windows PowerShell version
- `npm run validate:keys` - Comprehensive API key validation
- `npm run validate:keys:quick` - Quick validation without testing
- `npm run test:api-keys` - Alias for key validation
- `npm run security:check-keys` - Complete security audit

## 🛡️ Security Features

### **1. Secure Storage**
- **Location**: `.env.local` (automatically added to `.gitignore`)
- **Encryption**: Keys are validated and securely stored
- **Permissions**: Windows ACL permissions for enhanced security
- **Backup**: Automatic backup of existing configurations

### **2. Input Security**
- **Hidden Input**: Passwords/keys are never displayed in terminal
- **Pattern Validation**: Format validation for each API provider
- **Sanitization**: Input cleaning and validation
- **Timeout Protection**: Automatic timeout for security

### **3. API Testing**
- **Live Validation**: Real-time API endpoint testing
- **Service Coverage**: OpenAI, Google, NewsAPI, YouTube, Anthropic
- **Error Handling**: Detailed error reporting and recovery
- **Rate Limiting**: Respectful API usage patterns

## 📋 Supported APIs

### **OpenAI** (`OPENAI_API_KEY`)
- **Format**: `sk-proj-...` or `sk-...`
- **Test Endpoint**: `/models`
- **Required For**: GPT models, embeddings, completions

### **Google AI** (`GOOGLE_AI_API_KEY`)
- **Format**: 39-character alphanumeric
- **Test Endpoint**: Gemini Pro model
- **Required For**: Google AI services, Gemini

### **NewsAPI** (`NEWS_API_KEY`)
- **Format**: 32-character hexadecimal
- **Test Endpoint**: `/everything`
- **Required For**: News data fetching

### **YouTube** (`YOUTUBE_API_KEY`)
- **Format**: 39-character alphanumeric
- **Test Endpoint**: `/search`
- **Required For**: YouTube content integration

### **Anthropic** (`ANTHROPIC_API_KEY`)
- **Format**: `sk-ant-...`
- **Test Endpoint**: `/messages`
- **Required For**: Claude AI models

## 🔍 Validation Process

### **1. Format Validation**
```javascript
// Each API key is validated against specific patterns
const patterns = {
  OPENAI_API_KEY: /^sk-(proj-)?[a-zA-Z0-9]{32,}$/,
  GOOGLE_AI_API_KEY: /^[a-zA-Z0-9-_]{39}$/,
  NEWS_API_KEY: /^[a-fA-F0-9]{32}$/,
  YOUTUBE_API_KEY: /^[a-zA-Z0-9-_]{39}$/,
  ANTHROPIC_API_KEY: /^sk-ant-[a-zA-Z0-9-_]+$/
};
```

### **2. Live Testing**
```javascript
// Each key is tested against actual API endpoints
await testApiKey('OPENAI_API_KEY', openaiKey);
await testApiKey('GOOGLE_AI_API_KEY', googleKey);
// ... other services
```

### **3. Security Audit**
- Checks for exposed keys in code
- Validates `.env.local` permissions
- Ensures `.gitignore` includes sensitive files
- Verifies key format compliance

## 📁 File Structure

```
scripts/
├── secure-key-setup.js      # Main Node.js setup script
├── validate-api-keys.js     # Validation and testing
└── Setup-ApiKeys.ps1        # Windows PowerShell version

lib/
└── config.js               # Enhanced config with SecureKeyAgent

.env.local                  # Secure key storage (auto-created)
.gitignore                  # Updated with security entries
```

## 🔧 Configuration Details

### **Environment File Priority**
1. `.env.local` (highest priority - development)
2. `.env.development` (development-specific)
3. `.env` (default environment)

### **Security Masking**
```javascript
// Sensitive data is automatically masked in logs
const maskedConfig = {
  OPENAI_API_KEY: 'sk-proj-****...****',
  GOOGLE_AI_API_KEY: '****...****',
  // ... other keys
};
```

## 🚨 Security Best Practices

### **1. Key Management**
- ✅ **DO**: Use the SecureKeyAgent setup scripts
- ✅ **DO**: Store keys in `.env.local` only
- ✅ **DO**: Regularly validate and test keys
- ❌ **DON'T**: Hardcode keys in source code
- ❌ **DON'T**: Commit `.env.local` to version control
- ❌ **DON'T**: Share keys in chat/email

### **2. Development Workflow**
```bash
# 1. Initial setup
npm run setup:keys

# 2. Regular validation
npm run validate:keys

# 3. Before deployment
npm run security:check-keys

# 4. Testing
npm run test:api-keys
```

### **3. Troubleshooting**
```bash
# If keys are invalid
npm run validate:keys

# If API calls fail
npm run test:api-keys

# Complete security check
npm run security:check-keys
```

## 📊 Monitoring & Logging

### **Validation Reports**
```
=== API Key Validation Report ===
Generated: 2024-01-15T10:30:00.000Z

✅ OPENAI_API_KEY: Valid and tested
✅ GOOGLE_AI_API_KEY: Valid and tested
⚠️  NEWS_API_KEY: Valid format, test skipped
❌ YOUTUBE_API_KEY: Invalid format

Summary: 3/4 keys valid, 2/4 tested successfully
```

### **Security Hashes**
- Keys are hashed for secure logging
- No sensitive data in plain text
- Audit trail for key changes

## 🔄 Error Recovery

### **Common Issues & Solutions**

#### **Invalid API Key Format**
```bash
Error: Invalid format for OPENAI_API_KEY
Solution: Ensure key starts with 'sk-proj-' or 'sk-'
```

#### **API Test Failures**
```bash
Error: API test failed for NEWS_API_KEY
Solution: Check API quotas and rate limits
```

#### **Permission Issues (Windows)**
```bash
Error: Cannot set file permissions
Solution: Run PowerShell as Administrator
```

## 🚀 Advanced Usage

### **Custom Validation**
```javascript
// Add custom API providers
const customPatterns = {
  CUSTOM_API_KEY: /^custom-[a-zA-Z0-9]{20}$/
};
```

### **Batch Operations**
```bash
# Validate all keys quickly
npm run validate:keys:quick

# Full security audit
npm run security:check-keys
```

## 📞 Support & Troubleshooting

### **Debug Mode**
```bash
DEBUG=true npm run setup:keys
DEBUG=true npm run validate:keys
```

### **Log Files**
- Setup logs: `logs/secure-key-setup.log`
- Validation logs: `logs/api-validation.log`
- Security audit: `logs/security-audit.log`

### **Common Commands**
```bash
# Reset all keys
npm run setup:keys

# Check specific key
node scripts/validate-api-keys.js --key OPENAI_API_KEY

# Force re-setup
rm .env.local && npm run setup:keys
```

---

## 🎯 Mission Accomplished! 

The **SecureKeyAgent** provides enterprise-grade security for your AI News Dashboard. Your API keys are now:

- ✅ **Securely stored** in `.env.local`
- ✅ **Format validated** against provider standards
- ✅ **Live tested** for functionality
- ✅ **Protected** with proper permissions
- ✅ **Monitored** with comprehensive reporting

**Ready for production deployment!** 🚀
