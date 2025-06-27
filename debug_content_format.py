import sys
import os
import json
import yaml

# Add agents directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents'))

from rse_github_integrator import RSEContent, ContentType, RSEGitHubIntegrator, GitHubConfig

try:
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
    
    # Create test content
    test_content = RSEContent(
        title="Test Content Formatting",
        summary="Test summary for content formatting that meets the minimum character requirement for proper validation",
        domain="Data Science",
        keywords=["test", "formatting"],
        content="# Test Content\n\nThis is test content.",
        author="Test Author",
        content_type=ContentType.JSON
    )
    
    print("Testing JSON formatting...")
    # Test JSON formatting
    json_content = integrator.format_content(test_content)
    print(f"JSON content length: {len(json_content)}")
    
    try:
        json_data = json.loads(json_content)
        print("✓ JSON parsing successful")
        
        if 'metadata' in json_data:
            print("✓ 'metadata' found in JSON")
        else:
            print("✗ 'metadata' NOT found in JSON")
            
        if 'title' in json_data:
            print("✓ 'title' found in JSON")
        else:
            print("✗ 'title' NOT found in JSON")
            
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing failed: {e}")
    
    print("\nTesting Markdown formatting...")
    # Test Markdown formatting
    test_content.content_type = ContentType.MARKDOWN
    md_content = integrator.format_content(test_content)
    print(f"Markdown content length: {len(md_content)}")
    
    if md_content.startswith('---'):
        print("✓ Markdown starts with '---'")
    else:
        print("✗ Markdown does NOT start with '---'")
        print(f"Actually starts with: '{md_content[:20]}...'")
        
    if 'title:' in md_content:
        print("✓ 'title:' found in Markdown")
    else:
        print("✗ 'title:' NOT found in Markdown")
        
    if '---' in md_content:
        print("✓ '---' found in Markdown")
    else:
        print("✗ '---' NOT found in Markdown")
    
    print("\nTesting YAML formatting...")
    # Test YAML formatting
    test_content.content_type = ContentType.YAML
    yaml_content = integrator.format_content(test_content)
    print(f"YAML content length: {len(yaml_content)}")
    
    try:
        yaml_data = yaml.safe_load(yaml_content)
        print("✓ YAML parsing successful")
        
        if 'rse_content' in yaml_data:
            print("✓ 'rse_content' found in YAML")
        else:
            print("✗ 'rse_content' NOT found in YAML")
            print(f"YAML keys: {list(yaml_data.keys()) if isinstance(yaml_data, dict) else 'Not a dict'}")
            
    except yaml.YAMLError as e:
        print(f"✗ YAML parsing failed: {e}")
    
    print("\n✓ All content formatting tests completed successfully!")
    
except Exception as e:
    print(f"✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()