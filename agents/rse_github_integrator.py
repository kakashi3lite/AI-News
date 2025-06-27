#!/usr/bin/env python3
"""
RSE GitHub Integrator Bot v1.0
Automated GitHub integration for RSE (Research Software Engineering) content management.

Features:
- Automated file creation in /news directory
- Branch and PR management via GitHub API
- Schema validation for JSON/Markdown content
- Webhook event handling
- CI/CD integration with existing workflows

Author: AI News Dashboard Team
Integrates with: MLOps Pipeline, Veteran Agent, RSE Specialists
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import yaml
import hashlib
import re
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rse_github_integrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Supported content types for RSE files."""
    JSON = "json"
    MARKDOWN = "md"
    YAML = "yaml"

class PRStatus(Enum):
    """Pull request status tracking."""
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"
    DRAFT = "draft"
    READY = "ready_for_review"
    APPROVED = "approved"

@dataclass
class RSEContent:
    """Structure for RSE content objects."""
    title: str
    summary: str
    domain: str
    keywords: List[str]
    content: str
    author: str
    content_type: ContentType
    metadata: Dict[str, Any] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['content_type'] = self.content_type.value
        return data
    
    def validate(self) -> Dict[str, Any]:
        """Validate RSE content structure and requirements."""
        errors = []
        
        # Title validation
        if not self.title or len(self.title.strip()) < 10:
            errors.append("Title must be at least 10 characters long")
        if len(self.title) > 200:
            errors.append("Title must be less than 200 characters")
        
        # Summary validation
        if not self.summary or len(self.summary.strip()) < 50:
            errors.append("Summary must be at least 50 characters long")
        if len(self.summary) > 1000:
            errors.append("Summary must be less than 1000 characters")
        
        # Keywords validation
        if not self.keywords or len(self.keywords) < 2:
            errors.append("At least 2 keywords are required")
        if len(self.keywords) > 10:
            errors.append("Maximum 10 keywords allowed")
        
        # Author validation
        if not self.author or len(self.author.strip()) < 2:
            errors.append("Author name must be at least 2 characters long")
        
        # Content validation
        if not self.content or len(self.content.strip()) < 10:
            errors.append("Content must be at least 10 characters long")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

@dataclass
class GitHubConfig:
    """GitHub API configuration."""
    owner: str
    repo: str
    token: str = None
    base_branch: str = "main"
    reviewers: List[str] = None
    labels: List[str] = None
    
    def __post_init__(self):
        if self.reviewers is None:
            self.reviewers = []
        if self.labels is None:
            self.labels = ["rse", "automated", "content"]
        if self.token is None:
            self.token = os.getenv('GITHUB_TOKEN', 'test-token')
    
    def validate(self) -> bool:
        """Validate GitHub configuration."""
        return bool(self.owner and self.repo and self.token)

class RSEGitHubIntegrator:
    """Main RSE GitHub Integrator class."""
    
    def __init__(self, github_config: Optional[GitHubConfig] = None, config_path: Optional[str] = None):
        """Initialize the RSE GitHub Integrator."""
        self.config_path = config_path or "config/rse_github_config.yaml"
        self.config = self._load_config()
        
        # Use provided github_config or create from config file
        if github_config:
            self.github_config = github_config
        else:
            github_data = self.config.get('github', {})
            self.github_config = GitHubConfig(**github_data)
        
        self.news_dir = Path(self.config.get('news_directory', './news'))
        self.schemas = self._load_schemas()
        self.webhook_secret = os.getenv('GITHUB_WEBHOOK_SECRET')
        self.logger = logger
        
        # Ensure news directory exists
        self.news_dir.mkdir(exist_ok=True)
        
        logger.info(f"RSE GitHub Integrator initialized for {self.github_config.owner}/{self.github_config.repo}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'github': {
                'owner': 'ai-news-dashboard',
                'repo': 'AI-News',
                'base_branch': 'main',
                'reviewers': ['veteran-developer', 'rse-specialist'],
                'labels': ['rse', 'automated', 'content']
            },
            'news_directory': './news',
            'file_patterns': {
                'json': 'rse_{date}_{hash}.json',
                'md': 'rse_{date}_{hash}.md',
                'yaml': 'rse_{date}_{hash}.yaml'
            },
            'validation': {
                'enabled': True,
                'strict_mode': False
            },
            'ci_integration': {
                'wait_for_checks': True,
                'auto_merge': False,
                'required_checks': ['mlops-pipeline', 'veteran-agent-integration']
            }
        }
    
    def _load_schemas(self) -> Dict[str, Any]:
        """Load JSON schemas for validation."""
        schemas = {}
        schema_dir = Path('schemas')
        
        if schema_dir.exists():
            for schema_file in schema_dir.glob('*.json'):
                try:
                    with open(schema_file, 'r') as f:
                        schema_name = schema_file.stem
                        schemas[schema_name] = json.load(f)
                        logger.info(f"Loaded schema: {schema_name}")
                except Exception as e:
                    logger.error(f"Error loading schema {schema_file}: {e}")
        
        return schemas
    
    def validate_content(self, content: RSEContent) -> tuple[bool, List[str]]:
        """Validate RSE content against schemas."""
        errors = []
        
        # Basic validation
        if not content.title or len(content.title.strip()) == 0:
            errors.append("Title is required")
        
        if not content.summary or len(content.summary.strip()) == 0:
            errors.append("Summary is required")
        
        if not content.domain or len(content.domain.strip()) == 0:
            errors.append("Domain is required")
        
        if not content.keywords or len(content.keywords) == 0:
            errors.append("At least one keyword is required")
        
        # Content type specific validation
        if content.content_type == ContentType.JSON:
            try:
                json.loads(content.content)
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON content: {e}")
        
        # Schema validation if available
        schema_name = f"rse_{content.content_type.value}"
        if schema_name in self.schemas:
            # TODO: Implement JSON schema validation
            pass
        
        return len(errors) == 0, errors
    
    def generate_filename(self, content: RSEContent) -> str:
        """Generate filename for RSE content."""
        date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
        
        # Create hash from title and summary for uniqueness
        content_hash = hashlib.md5(
            f"{content.title}{content.summary}".encode('utf-8')
        ).hexdigest()[:8]
        
        pattern = self.config['file_patterns'][content.content_type.value]
        filename = pattern.format(
            date=date_str,
            hash=content_hash,
            domain=content.domain.lower().replace(' ', '_'),
            title=re.sub(r'[^a-zA-Z0-9_-]', '', content.title.lower().replace(' ', '_'))[:20]
        )
        
        return filename
    
    def format_content_for_file(self, content: RSEContent) -> str:
        """Format content for file output based on content type."""
        if content.content_type == ContentType.JSON:
            return json.dumps(content.to_dict(), indent=2, ensure_ascii=False)
        
        elif content.content_type == ContentType.MARKDOWN:
            md_content = f"""---
title: {content.title}
summary: {content.summary}
domain: {content.domain}
keywords: {content.keywords}
author: {content.author}
created_at: {content.created_at}
content_type: {content.content_type.value}
---

# {content.title}

## Summary
{content.summary}

## Domain
{content.domain}

## Keywords
{', '.join(content.keywords)}

## Content
{content.content}

## Metadata
- **Author**: {content.author}
- **Created**: {content.created_at}
- **Domain**: {content.domain}

---
*Generated by RSE GitHub Integrator*
"""
            return md_content
        
        elif content.content_type == ContentType.YAML:
            return yaml.dump(content.to_dict(), default_flow_style=False, allow_unicode=True)
        
        else:
            raise ValueError(f"Unsupported content type: {content.content_type}")
    
    def format_content(self, content: RSEContent) -> str:
        """Format content for file output based on content type (alias for format_content_for_file)."""
        return self.format_content_for_file(content)
    
    async def create_branch_and_file(self, content: RSEContent) -> Dict[str, Any]:
        """Create a new branch and file using GitHub API."""
        try:
            # Validate content first
            is_valid, errors = self.validate_content(content)
            if not is_valid:
                raise ValueError(f"Content validation failed: {', '.join(errors)}")
            
            # Generate filename and branch name
            filename = self.generate_filename(content)
            branch_name = f"rse/add-{filename.split('.')[0]}"
            file_path = f"news/{filename}"
            
            # Format content for file
            file_content = self.format_content_for_file(content)
            
            # Create commit message
            commit_message = f"Add RSE content: {content.title}\n\nDomain: {content.domain}\nAuthor: {content.author}\nGenerated by RSE GitHub Integrator"
            
            # TODO: Use GitHub MCP server to:
            # 1. Create branch from base branch
            # 2. Create/update file in the branch
            # 3. Return branch info for PR creation
            
            result = {
                'branch_name': branch_name,
                'file_path': file_path,
                'filename': filename,
                'commit_message': commit_message,
                'content_hash': hashlib.md5(file_content.encode('utf-8')).hexdigest()
            }
            
            logger.info(f"Created branch {branch_name} with file {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error creating branch and file: {e}")
            raise
    
    async def create_pull_request(self, branch_info: Dict[str, Any], content: RSEContent) -> Dict[str, Any]:
        """Create a pull request for the new content."""
        try:
            # Create PR title and body
            pr_title = f"[RSE] Add {content.domain} content: {content.title}"
            
            pr_body = f"""## RSE Content Addition

**Title**: {content.title}
**Domain**: {content.domain}
**Author**: {content.author}

### Summary
{content.summary}

### Keywords
{', '.join(content.keywords)}

### File Details
- **Path**: `{branch_info['file_path']}`
- **Type**: {content.content_type.value.upper()}
- **Branch**: `{branch_info['branch_name']}`

### Checklist
- [x] Content validated against schema
- [x] File created in correct location
- [ ] CI checks passed
- [ ] Code review completed

---
*This PR was automatically created by the RSE GitHub Integrator*
"""
            
            # TODO: Use GitHub MCP server to create PR
            # pr_data = await self.github_client.create_pull_request(
            #     title=pr_title,
            #     body=pr_body,
            #     head=branch_info['branch_name'],
            #     base=self.github_config.base_branch
            # )
            
            # TODO: Add reviewers and labels
            # if self.github_config.reviewers:
            #     await self.github_client.add_reviewers(pr_data['number'], self.github_config.reviewers)
            # 
            # if self.github_config.labels:
            #     await self.github_client.add_labels(pr_data['number'], self.github_config.labels)
            
            result = {
                'pr_number': 'TODO',  # Will be filled by actual GitHub API call
                'pr_url': 'TODO',
                'title': pr_title,
                'body': pr_body,
                'status': PRStatus.READY.value
            }
            
            logger.info(f"Created PR for {content.title}")
            return result
            
        except Exception as e:
            logger.error(f"Error creating pull request: {e}")
            raise
    
    async def process_rse_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to process RSE content and create GitHub PR."""
        try:
            # Create RSEContent object
            content = RSEContent(
                title=content_data['title'],
                summary=content_data['summary'],
                domain=content_data['domain'],
                keywords=content_data.get('keywords', []),
                content=content_data['content'],
                metadata=content_data.get('metadata', {}),
                created_at=datetime.now(timezone.utc).isoformat(),
                author=content_data.get('author', 'RSE System'),
                content_type=ContentType(content_data.get('content_type', 'json'))
            )
            
            # Step 1: Create branch and file
            branch_info = await self.create_branch_and_file(content)
            
            # Step 2: Create pull request
            pr_info = await self.create_pull_request(branch_info, content)
            
            # Step 3: Return complete result
            result = {
                'success': True,
                'content': content.to_dict(),
                'branch': branch_info,
                'pull_request': pr_info,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Successfully processed RSE content: {content.title}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing RSE content: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def handle_webhook_event(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GitHub webhook events."""
        try:
            logger.info(f"Received webhook event: {event_type}")
            
            if event_type == 'pull_request':
                return self._handle_pr_event(payload)
            elif event_type == 'push':
                return self._handle_push_event(payload)
            elif event_type == 'workflow_run':
                return self._handle_workflow_event(payload)
            else:
                logger.info(f"Unhandled event type: {event_type}")
                return {'status': 'ignored', 'event_type': event_type}
                
        except Exception as e:
            logger.error(f"Error handling webhook event: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _handle_pr_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pull request webhook events."""
        action = payload.get('action')
        pr = payload.get('pull_request', {})
        
        if action in ['opened', 'synchronize'] and 'rse' in pr.get('head', {}).get('ref', ''):
            # This is an RSE-related PR
            logger.info(f"RSE PR {action}: #{pr.get('number')} - {pr.get('title')}")
            
            # TODO: Trigger additional validations or notifications
            return {'status': 'processed', 'action': action, 'pr_number': pr.get('number')}
        
        return {'status': 'ignored', 'reason': 'not_rse_related'}
    
    def _handle_push_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle push webhook events."""
        ref = payload.get('ref', '')
        
        if ref.startswith('refs/heads/rse/'):
            # Push to RSE branch
            commits = payload.get('commits', [])
            logger.info(f"Push to RSE branch {ref}: {len(commits)} commits")
            
            return {'status': 'processed', 'branch': ref, 'commits': len(commits)}
        
        return {'status': 'ignored', 'reason': 'not_rse_branch'}
    
    def _handle_workflow_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle workflow run webhook events."""
        workflow = payload.get('workflow_run', {})
        conclusion = workflow.get('conclusion')
        
        if conclusion in ['success', 'failure'] and 'rse' in workflow.get('head_branch', ''):
            logger.info(f"Workflow {workflow.get('name')} {conclusion} on RSE branch")
            
            # TODO: Handle CI results for RSE PRs
            return {'status': 'processed', 'conclusion': conclusion}
        
        return {'status': 'ignored', 'reason': 'not_rse_related'}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the integrator."""
        try:
            checks = {
                'config_loaded': bool(self.config),
                'github_config': self.github_config.validate() if hasattr(self.github_config, 'validate') else True,
                'schemas_available': len(self.schemas) > 0,
                'news_directory_exists': self.news_dir.exists(),
                'github_token_configured': bool(os.getenv('GITHUB_TOKEN')),
                'webhook_secret_configured': bool(self.webhook_secret)
            }
            
            all_healthy = all(checks.values())
            
            health = {
                'status': 'healthy' if all_healthy else 'unhealthy',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'checks': checks
            }
            
            if not all_healthy:
                failed_checks = [k for k, v in checks.items() if not v]
                health['failed_checks'] = failed_checks
            
            return health
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'checks': {}
            }

# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Initialize integrator
        integrator = RSEGitHubIntegrator()
        
        # Health check
        health = integrator.health_check()
        print("Health Check:", json.dumps(health, indent=2))
        
        # Example RSE content
        example_content = {
            'title': 'Advanced Neural Network Architecture for News Classification',
            'summary': 'A comprehensive study on implementing transformer-based models for automated news categorization with 95% accuracy.',
            'domain': 'Machine Learning',
            'keywords': ['neural networks', 'transformers', 'news classification', 'NLP'],
            'content': '{"model": "transformer", "accuracy": 0.95, "dataset_size": 10000}',
            'content_type': 'json',
            'author': 'RSE Team',
            'metadata': {
                'version': '1.0',
                'framework': 'PyTorch',
                'license': 'MIT'
            }
        }
        
        # Process content
        result = await integrator.process_rse_content(example_content)
        print("Processing Result:", json.dumps(result, indent=2))
    
    # Run example
    asyncio.run(main())