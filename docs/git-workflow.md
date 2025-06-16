# Git Workflow Guide

## Overview

Simple Git workflow for the AI News Dashboard project.

## Quick Setup

```bash
# Clone and setup
git clone https://github.com/username/ai-news-dashboard.git
cd ai-news-dashboard
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

## Workflow

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Make changes and commit**: `git commit -m "feat: your changes"`
3. **Push and create PR**: `git push origin feature/your-feature`
4. **Merge after review**

## Branch Types

- `main` - Production ready code
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Emergency fixes

## Commit Messages

Use conventional commits:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation
- `chore:` - Maintenance

## Getting Help

Refer to the main README.md for project setup and contribution guidelines.