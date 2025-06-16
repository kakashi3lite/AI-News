# Contributing to AI News Dashboard

Welcome! We're excited to have you contribute to this AI-powered news platform.

## Quick Start

### Prerequisites
- Node.js 18+
- Git 2.25+
- PostgreSQL 13+ (or SQLite for development)

### Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/ai-news-dashboard.git
cd ai-news-dashboard

# 2. Install dependencies
npm install

# 3. Setup environment
cp .env.example .env.local
# Edit .env.local with your configuration

# 4. Start development
npm run dev
```

## Development Workflow

### Making Changes

```bash
# 1. Create a feature branch
git checkout -b feature/your-feature-name

# 2. Make your changes and test
npm test
npm run lint

# 3. Commit with conventional format
git commit -m "feat: add new feature"

# 4. Push and create PR
git push origin feature/your-feature-name
```

## Code Standards

### Commit Messages
Use conventional commits:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code formatting
- `refactor:` - Code restructuring
- `test:` - Adding tests
- `chore:` - Maintenance tasks

### Code Quality
- Follow ESLint configuration
- Write tests for new features
- Maintain 80%+ test coverage
- Use TypeScript for type safety
## Pull Request Process

### Before Submitting

```bash
# 1. Update your branch
git pull origin main
git rebase main

# 2. Run quality checks
npm test
npm run lint
npm run build
```

### PR Requirements
- Clear description of changes
- Tests for new features
- Documentation updates if needed
- No breaking changes without discussion

## Testing

```bash
# Run all tests
npm test

# Run specific test suites
npm run test:unit
npm run test:integration
npm run test:e2e

# Check test coverage
npm run test:coverage
```

## Documentation

- Update relevant documentation for new features
- Keep README.md current
- Add JSDoc comments for new functions
- Update API documentation if needed

## Getting Help

- Check existing issues and discussions
- Join our community discussions
- Review the documentation in `/docs`
- Ask questions in pull requests

Thank you for contributing! 🙏