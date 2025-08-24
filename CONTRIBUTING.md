# Contributing to AI News Dashboard

🎯 **Welcome to the AI News Dashboard project!** We're excited to have you contribute to this context-aware AI news platform.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Git Workflow](#git-workflow)
- [Branching Strategy](#branching-strategy)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Security](#security)

## 🚀 Getting Started

### Prerequisites

- **Node.js** 18+ and npm 8+
- **Git** 2.25+
- **PostgreSQL** 13+ (or SQLite for development)
- **Docker** (optional, for containerized development)

### Initial Setup

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/ai-news-dashboard.git
cd ai-news-dashboard

# 3. Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/ai-news-dashboard.git

# 4. Install dependencies
npm install

# 5. Copy environment variables
cp .env.example .env.local

# 6. Configure your environment variables
# Edit .env.local with your API keys and database URL

# 7. Run initial setup
npm run setup

# 8. Start development server
npm run dev
```

## 🔄 Development Workflow

### Daily Development

```bash
# 1. Start your day by syncing with upstream
git checkout main
git pull upstream main
git push origin main

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make your changes
# ... code, test, commit ...

# 4. Push your branch
git push origin feature/your-feature-name

# 5. Create a Pull Request on GitHub
```

### Keeping Your Fork Updated

```bash
# Fetch latest changes from upstream
git fetch upstream

# Switch to main branch
git checkout main

# Merge upstream changes
git merge upstream/main

# Push updates to your fork
git push origin main
```

## 🌿 Branching Strategy

We follow a **Git Flow** inspired branching model:

### Branch Types

| Branch Type | Naming Convention | Purpose | Base Branch |
|-------------|-------------------|---------|-------------|
| **Main** | `main` | Production-ready code | - |
| **Develop** | `develop` | Integration branch | `main` |
| **Feature** | `feature/description` | New features | `develop` |
| **Bugfix** | `bugfix/description` | Bug fixes | `develop` |
| **Hotfix** | `hotfix/description` | Critical production fixes | `main` |
| **Release** | `release/v1.2.3` | Release preparation | `develop` |
| **Chore** | `chore/description` | Maintenance tasks | `develop` |

### Branch Naming Examples

```bash
# Features
feature/arc-search-voice-support
feature/ai-skill-orchestrator
feature/social-recommendations

# Bug fixes
bugfix/search-overlay-mobile-layout
bugfix/context-memory-leak

# Hotfixes
hotfix/security-vulnerability-fix
hotfix/critical-performance-issue

# Chores
chore/update-dependencies
chore/improve-test-coverage
chore/refactor-api-endpoints
```

### Branch Lifecycle

```bash
# Creating a feature branch
git checkout develop
git pull upstream develop
git checkout -b feature/amazing-new-feature

# Working on the feature
git add .
git commit -m "feat: add amazing new feature"
git push origin feature/amazing-new-feature

# Creating a Pull Request
# - Target: develop branch
# - Include: description, screenshots, testing notes

# After PR approval and merge
git checkout develop
git pull upstream develop
git branch -d feature/amazing-new-feature
git push origin --delete feature/amazing-new-feature
```

## 📝 Commit Guidelines

We use **Conventional Commits** for consistent commit messages:

### Commit Message Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Commit Types

| Type | Description | Example |
|------|-------------|----------|
| `feat` | New feature | `feat(search): add voice search support` |
| `fix` | Bug fix | `fix(api): resolve memory leak in news fetcher` |
| `docs` | Documentation | `docs: update API documentation` |
| `style` | Code style changes | `style: format code with prettier` |
| `refactor` | Code refactoring | `refactor(components): extract common hooks` |
| `perf` | Performance improvements | `perf(search): optimize search algorithm` |
| `test` | Adding/updating tests | `test(api): add integration tests for news API` |
| `chore` | Maintenance tasks | `chore: update dependencies` |
| `ci` | CI/CD changes | `ci: add automated testing workflow` |
| `build` | Build system changes | `build: configure webpack optimization` |

### Commit Examples

```bash
# Good commits
git commit -m "feat(search): implement Arc-style search overlay with voice support"
git commit -m "fix(api): handle rate limiting in news aggregation"
git commit -m "docs(contributing): add Git workflow guidelines"
git commit -m "perf(context): optimize context prediction algorithm"

# Commits with body and footer
git commit -m "feat(ai): add skill orchestrator

Implement AI skill management system with:
- Dynamic skill loading
- Context-aware suggestions
- Performance monitoring

Closes #123
Breaking-change: API endpoint structure changed"
```

### Commit Best Practices

- **Use imperative mood**: "Add feature" not "Added feature"
- **Keep first line under 50 characters**
- **Capitalize first letter**
- **No period at the end of subject line**
- **Use body to explain what and why, not how**
- **Reference issues and PRs when relevant**

## 🔀 Pull Request Process

### Before Creating a PR

```bash
# 1. Ensure your branch is up to date
git checkout develop
git pull upstream develop
git checkout your-feature-branch
git rebase develop

# 2. Run tests and linting
npm run test
npm run lint
npm run type-check

# 3. Build the project
npm run build

# 4. Run security audit
npm audit
```

### PR Template

When creating a PR, use this template:

```markdown
## 🎯 Description

Brief description of changes and motivation.

## 🔄 Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## 🧪 Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] E2E tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## 📸 Screenshots/Videos

<!-- Add screenshots or videos for UI changes -->

## 📋 Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Code is commented where necessary
- [ ] Documentation updated
- [ ] No new warnings introduced
- [ ] Tests added/updated for changes
- [ ] All CI checks pass

## 🔗 Related Issues

Closes #issue_number
Related to #issue_number

## 🚀 Deployment Notes

<!-- Any special deployment considerations -->
```

### PR Review Process

1. **Automated Checks**: All CI/CD checks must pass
2. **Code Review**: At least one maintainer approval required
3. **Testing**: Manual testing for UI changes
4. **Documentation**: Ensure docs are updated
5. **Performance**: Check for performance regressions

### PR Merge Strategy

- **Squash and Merge**: For feature branches (default)
- **Merge Commit**: For release branches
- **Rebase and Merge**: For small, clean commits

## 🎨 Code Standards

### TypeScript/JavaScript

```bash
# Linting
npm run lint
npm run lint:fix

# Type checking
npm run type-check

# Formatting
npm run format
```

### Code Style Rules

- **TypeScript**: Strict mode enabled
- **ESLint**: Airbnb configuration with custom rules
- **Prettier**: Consistent code formatting
- **Import Order**: Absolute imports before relative
- **Naming**: camelCase for variables, PascalCase for components

### File Organization

```
src/
├── components/          # Reusable UI components
│   ├── ui/             # Basic UI primitives
│   └── features/       # Feature-specific components
├── hooks/              # Custom React hooks
├── utils/              # Utility functions
├── types/              # TypeScript type definitions
├── constants/          # Application constants
└── styles/             # Global styles and themes
```

## 🧪 Testing Requirements

### Test Coverage Requirements

- **Unit Tests**: 70% minimum coverage
- **Integration Tests**: Critical user flows
- **E2E Tests**: Main features and edge cases
- **Performance Tests**: Core web vitals validation

### Running Tests

```bash
# Unit tests
npm run test
npm run test:watch
npm run test:coverage

# Integration tests
npm run test:integration

# E2E tests
npm run test:e2e
npm run test:e2e:open

# Performance tests
npm run test:performance

# All tests
npm run test:all
```

### Test Structure

```javascript
// Component test example
import { render, screen, fireEvent } from '@testing-library/react';
import { ArcSearchOverlay } from '../ArcSearchOverlay';

describe('ArcSearchOverlay', () => {
  it('should open when Ctrl+K is pressed', () => {
    render(<ArcSearchOverlay />);
    
    fireEvent.keyDown(document, {
      key: 'k',
      ctrlKey: true
    });
    
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });
});
```

## 📚 Documentation

### Documentation Requirements

- **README**: Keep main README updated
- **API Docs**: Document all API endpoints
- **Component Docs**: JSDoc for all components
- **Architecture Docs**: Update for significant changes
- **Changelog**: Update for each release

### Documentation Standards

```javascript
/**
 * Arc-style search overlay component with voice support
 * 
 * @param {Object} props - Component props
 * @param {boolean} props.isOpen - Whether the overlay is open
 * @param {Function} props.onClose - Callback when overlay closes
 * @param {string} props.placeholder - Search input placeholder
 * @returns {JSX.Element} Search overlay component
 * 
 * @example
 * <ArcSearchOverlay 
 *   isOpen={true}
 *   onClose={() => setIsOpen(false)}
 *   placeholder="Search news..."
 * />
 */
export function ArcSearchOverlay({ isOpen, onClose, placeholder }) {
  // Component implementation
}
```

## 🐛 Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
## 🐛 Bug Description

Clear description of the bug.

## 🔄 Steps to Reproduce

1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

## 🎯 Expected Behavior

What you expected to happen.

## 📸 Screenshots

Add screenshots if applicable.

## 🖥️ Environment

- OS: [e.g. macOS 12.0]
- Browser: [e.g. Chrome 95.0]
- Node.js: [e.g. 18.0.0]
- npm: [e.g. 8.0.0]

## 📋 Additional Context

Any other context about the problem.
```

### Feature Requests

```markdown
## 🚀 Feature Request

### Problem Description
What problem does this feature solve?

### Proposed Solution
Describe your proposed solution.

### Alternatives Considered
Describe alternatives you've considered.

### Additional Context
Any other context or screenshots.
```

## 🔒 Security

### Security Issues

**DO NOT** create public issues for security vulnerabilities.

Instead:
1. Email: security@ai-news-dashboard.com
2. Include: detailed description and steps to reproduce
3. Wait for acknowledgment before public disclosure

### Security Best Practices

- Never commit secrets or API keys
- Use environment variables for sensitive data
- Follow OWASP security guidelines
- Regular dependency updates
- Input validation and sanitization

## 🎉 Recognition

Contributors are recognized in:
- **README**: Contributors section
- **Changelog**: Release notes
- **GitHub**: Contributor graphs
- **Discord**: Contributor role

## 📞 Getting Help

- **GitHub Discussions**: Q&A and ideas
- **Discord**: Real-time community chat
- **Email**: maintainers@ai-news-dashboard.com
- **Documentation**: Comprehensive guides

---

**Thank you for contributing to AI News Dashboard! 🙏**

*Together, we're building the future of context-aware news consumption.*
## 🛠 Development Commands

```bash
npm run lint        # ESLint
npm test            # Node tests
pytest              # Python tests
```

### 🔖 Triage Labels
Use labels: `bug`, `feature`, `chore`, `docs` for issues and PRs.

### 📝 Commit & PR Conventions
- Follow Conventional Commits (`feat:`, `fix:`, `chore:`)
- Reference issues with `Closes #123`
