#!/bin/bash
#
# Git Setup Script for AI News Dashboard
# Configures Git hooks, aliases, and project-specific settings
#
# Usage: ./scripts/setup-git.sh [options]
# Options:
#   --force    Force overwrite existing configurations
#   --minimal  Install minimal configuration only
#   --help     Show this help message
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
FORCE=false
MINIMAL=false
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GITHOOKS_DIR="$PROJECT_ROOT/.githooks"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"

# Helper functions
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

print_step() {
    echo -e "${PURPLE}→ $1${NC}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --minimal)
            MINIMAL=true
            shift
            ;;
        --help)
            echo "Git Setup Script for AI News Dashboard"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --force    Force overwrite existing configurations"
            echo "  --minimal  Install minimal configuration only"
            echo "  --help     Show this help message"
            echo ""
            echo "This script will:"
            echo "  - Configure Git hooks for code quality"
            echo "  - Set up useful Git aliases"
            echo "  - Configure project-specific Git settings"
            echo "  - Install commit message templates"
            echo "  - Set up branch protection workflows"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if we're in a Git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a Git repository. Please run this script from the project root."
    exit 1
fi

print_header "AI News Dashboard Git Setup"
print_info "Project root: $PROJECT_ROOT"
print_info "Git hooks directory: $GITHOOKS_DIR"
print_info "Force mode: $FORCE"
print_info "Minimal mode: $MINIMAL"

# 1. Create .githooks directory if it doesn't exist
print_header "Setting up Git hooks directory"
if [ ! -d "$GITHOOKS_DIR" ]; then
    mkdir -p "$GITHOOKS_DIR"
    print_success "Created .githooks directory"
else
    print_info ".githooks directory already exists"
fi

# 2. Configure Git to use custom hooks directory
print_step "Configuring Git hooks path"
if git config core.hooksPath "$GITHOOKS_DIR" 2>/dev/null; then
    print_success "Git hooks path configured to $GITHOOKS_DIR"
else
    print_error "Failed to configure Git hooks path"
    exit 1
fi

# 3. Make hooks executable
print_step "Making hooks executable"
for hook in "$GITHOOKS_DIR"/*; do
    if [ -f "$hook" ] && [ ! -x "$hook" ]; then
        chmod +x "$hook"
        print_success "Made $(basename "$hook") executable"
    fi
done

# 4. Set up commit message template
print_header "Configuring commit message template"
if [ -f "$PROJECT_ROOT/.gitmessage" ]; then
    if git config commit.template "$PROJECT_ROOT/.gitmessage" 2>/dev/null; then
        print_success "Commit message template configured"
    else
        print_warning "Failed to configure commit message template"
    fi
else
    print_warning "Commit message template file not found"
fi

# 5. Configure useful Git aliases
print_header "Setting up Git aliases"

# Define aliases
declare -A aliases=(
    ["st"]="status -s"
    ["co"]="checkout"
    ["br"]="branch"
    ["ci"]="commit"
    ["ca"]="commit -a"
    ["cm"]="commit -m"
    ["cam"]="commit -am"
    ["unstage"]="reset HEAD --"
    ["last"]="log -1 HEAD"
    ["visual"]="!gitk"
    ["graph"]="log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
    ["lg"]="log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
    ["contributors"]="shortlog --summary --numbered"
    ["amend"]="commit --amend --no-edit"
    ["fixup"]="commit --fixup"
    ["squash"]="commit --squash"
    ["wip"]="commit -am 'WIP: work in progress'"
    ["unwip"]="reset HEAD~1"
    ["aliases"]="config --get-regexp alias"
    ["branches"]="branch -a"
    ["remotes"]="remote -v"
    ["tags"]="tag -l"
    ["stashes"]="stash list"
    ["conflicts"]="diff --name-only --diff-filter=U"
    ["resolve"]="add -u"
    ["abort"]="merge --abort"
    ["continue"]="rebase --continue"
    ["skip"]="rebase --skip"
    ["edit"]="rebase --edit-todo"
    ["root"]="rev-parse --show-toplevel"
    ["exec"]="!exec "
    ["feature"]="checkout -b feature/"
    ["bugfix"]="checkout -b bugfix/"
    ["hotfix"]="checkout -b hotfix/"
    ["release"]="checkout -b release/"
    ["develop"]="checkout develop"
    ["main"]="checkout main"
    ["master"]="checkout master"
    ["sync"]="!git fetch origin && git rebase origin/$(git branch --show-current)"
    ["update"]="!git fetch origin && git merge origin/$(git branch --show-current)"
    ["clean-branches"]="!git branch --merged | grep -v '\*\|main\|master\|develop' | xargs -n 1 git branch -d"
    ["recent"]="branch --sort=-committerdate"
    ["find"]="!git ls-files | grep -i"
    ["grep"]="grep -Ii"
    ["la"]="!git config -l | grep alias | cut -c 7-"
    ["save"]="!git add -A && git commit -m 'SAVEPOINT'"
    ["undo"]="reset HEAD~1 --mixed"
    ["res"]="!git reset --hard"
    ["reshard"]="reset --hard"
    ["type"]="cat-file -t"
    ["dump"]="cat-file -p"
    ["look"]="log --oneline --decorate"
    ["overview"]="log --all --since='2 weeks' --oneline --no-merges"
    ["recap"]="log --all --oneline --no-merges --author=$(git config user.email)"
    ["today"]="log --since=00:00:00 --all --no-merges --oneline --author=$(git config user.email)"
    ["changelog"]="log --oneline --no-merges"
)

# Install aliases
for alias_name in "${!aliases[@]}"; do
    alias_command="${aliases[$alias_name]}"
    
    # Check if alias already exists
    if git config --get alias."$alias_name" > /dev/null 2>&1; then
        if [ "$FORCE" = true ]; then
            git config alias."$alias_name" "$alias_command"
            print_success "Updated alias: $alias_name"
        else
            print_info "Alias '$alias_name' already exists (use --force to overwrite)"
        fi
    else
        git config alias."$alias_name" "$alias_command"
        print_success "Added alias: $alias_name"
    fi
done

# 6. Configure Git settings for the project
print_header "Configuring Git settings"

# Core settings
print_step "Setting core configurations"
git config core.autocrlf false
git config core.safecrlf true
git config core.filemode false
git config core.ignorecase false
git config core.precomposeunicode true
git config core.quotepath false
print_success "Core settings configured"

# Push settings
print_step "Setting push configurations"
git config push.default simple
git config push.followTags true
print_success "Push settings configured"

# Pull settings
print_step "Setting pull configurations"
git config pull.rebase true
git config rebase.autoStash true
print_success "Pull settings configured"

# Merge settings
print_step "Setting merge configurations"
git config merge.tool vimdiff
git config merge.conflictstyle diff3
print_success "Merge settings configured"

# Branch settings
print_step "Setting branch configurations"
git config branch.autosetupmerge always
git config branch.autosetuprebase always
print_success "Branch settings configured"

# Color settings
if [ "$MINIMAL" = false ]; then
    print_step "Setting color configurations"
    git config color.ui auto
    git config color.branch auto
    git config color.diff auto
    git config color.status auto
    git config color.grep auto
    git config color.interactive auto
    print_success "Color settings configured"
fi

# 7. Set up additional configurations for development
if [ "$MINIMAL" = false ]; then
    print_header "Setting up development configurations"
    
    # Configure diff and merge tools
    print_step "Configuring diff and merge tools"
    
    # Check for VS Code
    if command -v code > /dev/null 2>&1; then
        git config diff.tool vscode
        git config difftool.vscode.cmd 'code --wait --diff $LOCAL $REMOTE'
        git config merge.tool vscode
        git config mergetool.vscode.cmd 'code --wait $MERGED'
        print_success "VS Code configured as diff/merge tool"
    fi
    
    # Configure rerere (reuse recorded resolution)
    git config rerere.enabled true
    print_success "Rerere enabled for conflict resolution"
    
    # Configure help settings
    git config help.autocorrect 1
    print_success "Help autocorrect enabled"
    
    # Configure log settings
    git config log.date relative
    print_success "Log date format set to relative"
fi

# 8. Create useful Git scripts
print_header "Creating utility scripts"

# Create git-cleanup script
cat > "$SCRIPTS_DIR/git-cleanup.sh" << 'EOF'
#!/bin/bash
# Git cleanup script - removes merged branches and cleans up repository

set -e

echo "🧹 Cleaning up Git repository..."

# Fetch latest changes
echo "📡 Fetching latest changes..."
git fetch --all --prune

# Clean up merged branches
echo "🌿 Removing merged branches..."
git branch --merged | grep -v "\*\|main\|master\|develop" | xargs -n 1 git branch -d 2>/dev/null || true

# Clean up remote tracking branches
echo "🔗 Cleaning up remote tracking branches..."
git remote prune origin

# Clean up stale references
echo "🗑️  Cleaning up stale references..."
git gc --prune=now

# Show current status
echo "📊 Repository status:"
git status -s
echo "🌿 Remaining branches:"
git branch -a

echo "✅ Cleanup complete!"
EOF

chmod +x "$SCRIPTS_DIR/git-cleanup.sh"
print_success "Created git-cleanup.sh script"

# Create git-stats script
cat > "$SCRIPTS_DIR/git-stats.sh" << 'EOF'
#!/bin/bash
# Git statistics script - shows repository statistics

set -e

echo "📊 Git Repository Statistics"
echo "============================"
echo ""

# Repository info
echo "📁 Repository Information:"
echo "   Root: $(git rev-parse --show-toplevel)"
echo "   Branch: $(git branch --show-current)"
echo "   Remote: $(git remote get-url origin 2>/dev/null || echo 'No remote configured')"
echo ""

# Commit statistics
echo "📈 Commit Statistics:"
echo "   Total commits: $(git rev-list --all --count)"
echo "   Commits this month: $(git rev-list --since='1 month ago' --count HEAD)"
echo "   Commits this week: $(git rev-list --since='1 week ago' --count HEAD)"
echo "   Commits today: $(git rev-list --since='1 day ago' --count HEAD)"
echo ""

# Author statistics
echo "👥 Top Contributors (last 3 months):"
git shortlog -sn --since='3 months ago' | head -10
echo ""

# Branch information
echo "🌿 Branch Information:"
echo "   Total branches: $(git branch -a | wc -l)"
echo "   Local branches: $(git branch | wc -l)"
echo "   Remote branches: $(git branch -r | wc -l)"
echo ""

# Recent activity
echo "🕒 Recent Activity (last 10 commits):"
git log --oneline -10
echo ""

# File statistics
echo "📄 File Statistics:"
echo "   Total files: $(git ls-files | wc -l)"
echo "   Largest files:"
git ls-tree -r -l HEAD | sort -k 4 -n | tail -5 | awk '{print "     " $4 " bytes - " $5}'
echo ""

# Repository size
echo "💾 Repository Size:"
echo "   .git directory: $(du -sh .git 2>/dev/null | cut -f1)"
echo "   Working directory: $(du -sh --exclude=.git . 2>/dev/null | cut -f1)"
EOF

chmod +x "$SCRIPTS_DIR/git-stats.sh"
print_success "Created git-stats.sh script"

# 9. Test Git hooks
print_header "Testing Git hooks"

if [ -f "$GITHOOKS_DIR/pre-commit" ]; then
    print_step "Testing pre-commit hook"
    if [ -x "$GITHOOKS_DIR/pre-commit" ]; then
        print_success "Pre-commit hook is executable"
    else
        print_warning "Pre-commit hook is not executable"
    fi
fi

if [ -f "$GITHOOKS_DIR/commit-msg" ]; then
    print_step "Testing commit-msg hook"
    if [ -x "$GITHOOKS_DIR/commit-msg" ]; then
        print_success "Commit-msg hook is executable"
    else
        print_warning "Commit-msg hook is not executable"
    fi
fi

if [ -f "$GITHOOKS_DIR/pre-push" ]; then
    print_step "Testing pre-push hook"
    if [ -x "$GITHOOKS_DIR/pre-push" ]; then
        print_success "Pre-push hook is executable"
    else
        print_warning "Pre-push hook is not executable"
    fi
fi

# 10. Create .gitattributes if it doesn't exist
print_header "Setting up .gitattributes"

if [ ! -f "$PROJECT_ROOT/.gitattributes" ] || [ "$FORCE" = true ]; then
    cat > "$PROJECT_ROOT/.gitattributes" << 'EOF'
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
EOF
    print_success "Created .gitattributes file"
else
    print_info ".gitattributes already exists (use --force to overwrite)"
fi

# 11. Final summary and next steps
print_header "Setup Complete!"

print_success "Git configuration has been successfully set up for AI News Dashboard"
echo ""
print_info "What was configured:"
echo "  ✓ Git hooks directory and executable permissions"
echo "  ✓ Commit message template"
echo "  ✓ Useful Git aliases ($(echo "${!aliases[@]}" | wc -w) aliases)"
echo "  ✓ Project-specific Git settings"
echo "  ✓ Development tools configuration"
echo "  ✓ Utility scripts for cleanup and statistics"
echo "  ✓ .gitattributes for proper file handling"
echo ""
print_info "Available commands:"
echo "  git aliases          - List all configured aliases"
echo "  ./scripts/git-cleanup.sh  - Clean up merged branches"
echo "  ./scripts/git-stats.sh    - Show repository statistics"
echo ""
print_info "Useful aliases to try:"
echo "  git st              - Short status"
echo "  git lg              - Pretty log graph"
echo "  git sync            - Sync with remote branch"
echo "  git clean-branches  - Remove merged branches"
echo "  git wip             - Quick work-in-progress commit"
echo "  git recent          - Show recent branches"
echo ""
print_info "Next steps:"
echo "  1. Test the setup with a small commit"
echo "  2. Review the Git hooks in .githooks/"
echo "  3. Customize aliases as needed"
echo "  4. Share this setup with your team"
echo ""
print_success "Happy coding! 🚀"