import sys
import os
import json

# Add agents directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents'))

from rse_github_integrator import RSEContent, ContentType, RSEGitHubIntegrator, GitHubConfig

# Create test content
test_content = RSEContent(
    title="Test Content Formatting",
    summary="Test summary for content formatting that is long enough to pass validation requirements",
    domain="Data Science",
    keywords=["test", "formatting"],
    content="# Test Content\n\nThis is test content.",
    author="Test Author",
    content_type=ContentType.JSON
)

# Create test integrator
github_config = GitHubConfig(
    owner="test-owner",
    repo="test-repo",
    token="test-token",
    base_branch="main"
)

integrator = RSEGitHubIntegrator(
    github_config=github_config,
    config_path="config.yaml"
)

# Test JSON formatting
json_content = integrator.format_content(test_content)
print("JSON Content:")
print(json_content)
print("\n" + "="*50 + "\n")

# Parse and check structure
json_data = json.loads(json_content)
print("JSON Structure:")
for key in json_data.keys():
    print(f"- {key}")

print("\nChecking for 'metadata' and 'title':")
print(f"'metadata' in json_data: {'metadata' in json_data}")
print(f"'title' in json_data: {'title' in json_data}")