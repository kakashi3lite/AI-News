# Git Workflow Guide

🔄 **Comprehensive Git workflow documentation for the AI News Dashboard project.**

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Setup](#repository-setup)
- [Branching Strategy](#branching-strategy)
- [Daily Workflow](#daily-workflow)
- [Feature Development](#feature-development)
- [Bug Fixes](#bug-fixes)
- [Hotfixes](#hotfixes)
- [Release Process](#release-process)
- [Code Review Process](#code-review-process)
- [Merge Strategies](#merge-strategies)
- [Git Best Practices](#git-best-practices)
- [Troubleshooting](#troubleshooting)
- [Advanced Git Techniques](#advanced-git-techniques)

## 🎯 Overview

Our Git workflow is designed to:
- Maintain code quality and stability
- Enable parallel development
- Facilitate code reviews
- Support continuous integration/deployment
- Provide clear project history

### Workflow Principles

1. **Main branch is always deployable**
2. **Feature branches for all new work**
3. **Code review required for all changes**
4. **Automated testing before merge**
5. **Clear commit messages and history**

## 🏗️ Repository Setup

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/username/ai-news-dashboard.git
cd ai-news-dashboard

# Configure Git (first time only)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Set up upstream remote (for forks)
git remote add upstream https://github.com/original-owner/ai-news-dashboard.git

# Verify remotes
git remote -v
# origin    https://github.com/username/ai-news-dashboard.git (fetch)
# origin    https://github.com/username/ai-news-dashboard.git (push)
# upstream  https://github.com/original-owner/ai-news-dashboard.git (fetch)
# upstream  https://github.com/original-owner/ai-news-dashboard.git (push)
```

### Git Configuration

```bash
# Global configuration
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input  # For cross-platform compatibility

# Project-specific configuration
git config core.hooksPath .githooks
git config commit.template .gitmessage
```

### Git Hooks Setup

```bash
# Make hooks executable
chmod +x .githooks/*

# Install pre-commit hooks
npm install husky --save-dev
npx husky install
npx husky add .husky/pre-commit "npm run lint && npm run test"
npx husky add .husky/commit-msg "npx commitlint --edit $1"
```

## 🌿 Branching Strategy

### Branch Types and Naming

```
main
├── develop
│   ├── feature/arc-search-voice-support
│   ├── feature/ai-skill-orchestrator
│   ├── bugfix/search-overlay-mobile
│   └── chore/update-dependencies
├── release/v1.1.0
└── hotfix/critical-security-fix
```

### Branch Naming Conventions

| Type | Format | Example | Description |
|------|--------|---------|-------------|
| Feature | `feature/description` | `feature/social-recommendations` | New features |
| Bugfix | `bugfix/description` | `bugfix/memory-leak-fix` | Bug fixes |
| Hotfix | `hotfix/description` | `hotfix/security-vulnerability` | Critical fixes |
| Release | `release/version` | `release/v1.2.0` | Release preparation |
| Chore | `chore/description` | `chore/update-docs` | Maintenance |
| Experiment | `experiment/description` | `experiment/new-ui-layout` | Experimental features |

### Branch Lifecycle

```bash
# 1. Create and switch to new branch
git checkout -b feature/amazing-feature develop

# 2. Work on the feature
# ... make changes, commit ...

# 3. Push branch to remote
git push -u origin feature/amazing-feature

# 4. Create Pull Request
# ... via GitHub interface ...

# 5. After merge, cleanup
git checkout develop
git pull upstream develop
git branch -d feature/amazing-feature
git push origin --delete feature/amazing-feature
```

## 📅 Daily Workflow

### Starting Your Day

```bash
# 1. Switch to main branch
git checkout main

# 2. Pull latest changes
git pull upstream main

# 3. Update your fork
git push origin main

# 4. Switch to develop
git checkout develop
git pull upstream develop
git push origin develop

# 5. Clean up merged branches
git branch --merged | grep -v "\*\|main\|develop" | xargs -n 1 git branch -d
```

### During Development

```bash
# Create feature branch
git checkout -b feature/new-feature develop

# Make changes and commit frequently
git add .
git commit -m "feat: add initial implementation"

# Push to remote regularly
git push origin feature/new-feature

# Keep branch updated with develop
git fetch upstream
git rebase upstream/develop
```

### End of Day

```bash
# Ensure all work is committed
git status
git add .
git commit -m "wip: end of day checkpoint"

# Push to remote
git push origin feature/new-feature

# Optional: Create draft PR for visibility
# ... via GitHub interface ...
```

## 🚀 Feature Development

### Complete Feature Workflow

```bash
# 1. Start from develop
git checkout develop
git pull upstream develop

# 2. Create feature branch
git checkout -b feature/arc-search-enhancement

# 3. Implement feature with atomic commits
git add src/components/ArcSearch.js
git commit -m "feat(search): add voice search button"

git add src/hooks/useVoiceSearch.js
git commit -m "feat(search): implement voice search hook"

git add tests/ArcSearch.test.js
git commit -m "test(search): add voice search tests"

# 4. Keep branch updated
git fetch upstream
git rebase upstream/develop

# 5. Push and create PR
git push origin feature/arc-search-enhancement
# Create PR via GitHub

# 6. Address review feedback
git add .
git commit -m "fix(search): address PR feedback"
git push origin feature/arc-search-enhancement

# 7. After approval, squash if needed
git rebase -i HEAD~3  # Interactive rebase to squash commits
git push --force-with-lease origin feature/arc-search-enhancement
```

### Feature Branch Best Practices

- **Small, focused features**: One feature per branch
- **Descriptive names**: Clear purpose from branch name
- **Regular updates**: Rebase with develop frequently
- **Atomic commits**: Each commit should be a logical unit
- **Tests included**: Add tests for new functionality
- **Documentation**: Update docs for user-facing changes

## 🐛 Bug Fixes

### Bug Fix Workflow

```bash
# 1. Create bugfix branch from develop
git checkout develop
git pull upstream develop
git checkout -b bugfix/search-overlay-mobile-layout

# 2. Reproduce and fix the bug
# ... investigate, fix, test ...

# 3. Commit with clear description
git add .
git commit -m "fix(search): resolve mobile layout issues in overlay

- Fix z-index conflicts on mobile devices
- Adjust responsive breakpoints for small screens
- Improve touch target sizes for mobile interaction

Fixes #123"

# 4. Add regression test
git add tests/SearchOverlay.mobile.test.js
git commit -m "test(search): add mobile layout regression tests"

# 5. Push and create PR
git push origin bugfix/search-overlay-mobile-layout
```

### Bug Fix Guidelines

- **Root cause analysis**: Understand why the bug occurred
- **Minimal changes**: Fix only what's necessary
- **Regression tests**: Prevent the bug from reoccurring
- **Clear documentation**: Explain the fix in commit message
- **Verification**: Test fix in multiple environments

## 🚨 Hotfixes

### Critical Hotfix Process

```bash
# 1. Create hotfix branch from main
git checkout main
git pull upstream main
git checkout -b hotfix/critical-security-vulnerability

# 2. Implement minimal fix
git add .
git commit -m "fix(security): patch XSS vulnerability in search input

Security fix for CVE-2024-XXXX
- Sanitize user input in search component
- Add input validation middleware
- Update security headers

Severity: Critical
Affected versions: 1.0.0 - 1.0.5"

# 3. Test thoroughly
npm run test
npm run test:security

# 4. Create PR targeting main
git push origin hotfix/critical-security-vulnerability
# Create PR to main branch

# 5. After merge to main, also merge to develop
git checkout develop
git pull upstream develop
git merge upstream/main
git push upstream develop

# 6. Tag the release
git checkout main
git pull upstream main
git tag -a v1.0.6 -m "Security hotfix release v1.0.6"
git push upstream v1.0.6
```

### Hotfix Criteria

- **Security vulnerabilities**
- **Critical bugs affecting all users**
- **Data loss or corruption issues**
- **Service outages**
- **Legal or compliance issues**

## 🏷️ Release Process

### Release Branch Workflow

```bash
# 1. Create release branch from develop
git checkout develop
git pull upstream develop
git checkout -b release/v1.1.0

# 2. Update version numbers
npm version 1.1.0 --no-git-tag-version
git add package.json package-lock.json
git commit -m "chore(release): bump version to 1.1.0"

# 3. Update changelog
# Edit CHANGELOG.md
git add CHANGELOG.md
git commit -m "docs(changelog): update for v1.1.0 release"

# 4. Final testing and bug fixes
# ... fix any release-blocking issues ...

# 5. Merge to main
git checkout main
git pull upstream main
git merge --no-ff release/v1.1.0
git push upstream main

# 6. Tag the release
git tag -a v1.1.0 -m "Release version 1.1.0

Features:
- Arc-style search enhancements
- AI skill orchestrator improvements
- Social features expansion

Bug fixes:
- Mobile layout issues
- Performance optimizations
- Security updates"
git push upstream v1.1.0

# 7. Merge back to develop
git checkout develop
git merge --no-ff release/v1.1.0
git push upstream develop

# 8. Clean up release branch
git branch -d release/v1.1.0
git push origin --delete release/v1.1.0
```

### Release Checklist

- [ ] All features complete and tested
- [ ] Version numbers updated
- [ ] Changelog updated
- [ ] Documentation updated
- [ ] Security audit passed
- [ ] Performance benchmarks met
- [ ] Browser compatibility tested
- [ ] Deployment scripts tested
- [ ] Rollback plan prepared

## 👀 Code Review Process

### Creating a Pull Request

```bash
# 1. Ensure branch is up to date
git fetch upstream
git rebase upstream/develop

# 2. Run pre-submission checks
npm run lint
npm run test
npm run build
npm run type-check

# 3. Push final changes
git push origin feature/branch-name

# 4. Create PR with template
# Use GitHub interface with PR template
```

### PR Review Guidelines

#### For Authors
- **Self-review first**: Review your own changes
- **Clear description**: Explain what and why
- **Screenshots**: Include for UI changes
- **Test coverage**: Ensure adequate testing
- **Documentation**: Update relevant docs
- **Small PRs**: Keep changes focused and reviewable

#### For Reviewers
- **Timely reviews**: Review within 24 hours
- **Constructive feedback**: Suggest improvements
- **Test the changes**: Pull and test locally
- **Check edge cases**: Consider error scenarios
- **Approve explicitly**: Use GitHub's review system

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are comprehensive
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance impact considered
- [ ] Accessibility requirements met
- [ ] Browser compatibility maintained
- [ ] Error handling is robust

## 🔀 Merge Strategies

### When to Use Each Strategy

#### Squash and Merge (Default)
```bash
# Use for: Feature branches with multiple commits
# Result: Single commit in target branch
git checkout develop
git merge --squash feature/branch-name
git commit -m "feat: implement amazing feature

Detailed description of the feature...

Closes #123"
```

#### Merge Commit
```bash
# Use for: Release branches, important milestones
# Result: Preserves branch history
git checkout main
git merge --no-ff release/v1.1.0
```

#### Rebase and Merge
```bash
# Use for: Clean, atomic commits
# Result: Linear history without merge commit
git checkout develop
git rebase feature/branch-name
```

### Merge Strategy Decision Tree

```
Is it a feature branch with messy commits?
├─ Yes → Squash and Merge
└─ No
   ├─ Is it a release or important milestone?
   │  ├─ Yes → Merge Commit
   │  └─ No
   │     ├─ Are commits clean and atomic?
   │     │  ├─ Yes → Rebase and Merge
   │     │  └─ No → Squash and Merge
```

## 📚 Git Best Practices

### Commit Message Guidelines

```bash
# Good commit messages
git commit -m "feat(search): add voice search support"
git commit -m "fix(api): handle rate limiting in news fetcher"
git commit -m "docs: update API documentation"
git commit -m "perf(context): optimize prediction algorithm"

# Commit with body and footer
git commit -m "feat(ai): implement skill orchestrator

Add AI skill management system with:
- Dynamic skill loading
- Context-aware suggestions
- Performance monitoring

Breaking-change: API endpoint structure changed
Closes #123
Reviewed-by: @reviewer"
```

### Commit Best Practices

1. **Atomic commits**: One logical change per commit
2. **Clear messages**: Describe what and why, not how
3. **Present tense**: "Add feature" not "Added feature"
4. **Imperative mood**: "Fix bug" not "Fixes bug"
5. **Reference issues**: Include issue numbers
6. **Breaking changes**: Clearly mark breaking changes

### Branch Management

```bash
# List all branches
git branch -a

# Delete merged branches
git branch --merged | grep -v "\*\|main\|develop" | xargs -n 1 git branch -d

# Prune remote tracking branches
git remote prune origin

# Clean up local repository
git gc --prune=now
```

### Keeping History Clean

```bash
# Interactive rebase to clean up commits
git rebase -i HEAD~3

# Amend last commit
git commit --amend

# Reset to previous commit (careful!)
git reset --soft HEAD~1

# Stash changes temporarily
git stash push -m "work in progress"
git stash pop
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Merge Conflicts

```bash
# When conflicts occur during merge/rebase
git status  # See conflicted files

# Edit files to resolve conflicts
# Look for conflict markers: <<<<<<<, =======, >>>>>>>

# After resolving conflicts
git add resolved-file.js
git commit  # For merge
# OR
git rebase --continue  # For rebase
```

#### Accidentally Committed to Wrong Branch

```bash
# Move commits to correct branch
git log --oneline -n 3  # Note commit hashes
git checkout correct-branch
git cherry-pick commit-hash

# Remove from wrong branch
git checkout wrong-branch
git reset --hard HEAD~1  # Remove last commit
```

#### Undo Last Commit

```bash
# Keep changes in working directory
git reset --soft HEAD~1

# Discard changes completely
git reset --hard HEAD~1

# Create new commit that undoes changes
git revert HEAD
```

#### Force Push Safely

```bash
# Safer than --force
git push --force-with-lease origin branch-name

# Check what would be overwritten
git push --dry-run --force-with-lease origin branch-name
```

#### Recover Lost Commits

```bash
# Find lost commits
git reflog

# Recover specific commit
git checkout commit-hash
git checkout -b recovery-branch
```

### Emergency Procedures

#### Rollback Production

```bash
# Quick rollback to previous tag
git checkout main
git reset --hard v1.0.5
git push --force-with-lease origin main

# Deploy previous version
npm run deploy:production
```

#### Fix Broken Main Branch

```bash
# Create fix branch from last known good commit
git checkout -b emergency-fix last-good-commit-hash

# Apply minimal fix
git add .
git commit -m "fix: emergency production fix"

# Force update main (with team coordination)
git checkout main
git reset --hard emergency-fix
git push --force-with-lease origin main
```

## 🎓 Advanced Git Techniques

### Interactive Rebase

```bash
# Rewrite commit history
git rebase -i HEAD~5

# Options in interactive mode:
# pick = use commit
# reword = use commit, but edit message
# edit = use commit, but stop for amending
# squash = use commit, but meld into previous commit
# fixup = like squash, but discard commit message
# drop = remove commit
```

### Git Bisect for Bug Hunting

```bash
# Start bisect session
git bisect start
git bisect bad  # Current commit is bad
git bisect good v1.0.0  # Known good commit

# Git will checkout middle commit
# Test and mark as good or bad
git bisect good  # or git bisect bad

# Continue until bug is found
git bisect reset  # End session
```

### Advanced Merging

```bash
# Merge with custom strategy
git merge -X theirs feature-branch  # Prefer their changes
git merge -X ours feature-branch    # Prefer our changes

# Merge specific files only
git checkout feature-branch -- path/to/file.js
git commit -m "merge: bring in specific file from feature"
```

### Git Hooks

```bash
# Pre-commit hook example (.git/hooks/pre-commit)
#!/bin/sh
npm run lint
npm run test
if [ $? -ne 0 ]; then
  echo "Tests failed. Commit aborted."
  exit 1
fi
```

### Git Aliases

```bash
# Useful aliases
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual '!gitk'
git config --global alias.graph 'log --oneline --graph --decorate --all'
```

## 📊 Git Workflow Metrics

### Tracking Workflow Health

```bash
# Branch age analysis
git for-each-ref --format='%(refname:short) %(committerdate)' refs/heads | sort -k2

# Commit frequency
git log --since="1 month ago" --pretty=format:"%an" | sort | uniq -c | sort -nr

# File change frequency
git log --name-only --pretty=format: | sort | uniq -c | sort -nr
```

### Workflow KPIs

- **Average PR size**: < 400 lines changed
- **PR review time**: < 24 hours
- **Branch lifetime**: < 1 week
- **Merge conflicts**: < 5% of PRs
- **Hotfix frequency**: < 1 per month
- **Test coverage**: > 80%
- **Build success rate**: > 95%

---

## 📞 Support

For Git workflow questions:
- **GitHub Discussions**: Workflow questions and best practices
- **Team Chat**: Real-time Git help
- **Documentation**: This guide and Git documentation
- **Training**: Git workshops and pair programming

---

**Happy coding! 🚀**

*Remember: Good Git practices lead to better collaboration and fewer headaches.*