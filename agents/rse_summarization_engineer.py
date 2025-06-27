#!/usr/bin/env python3
"""
RSE Summarization Engineer - AI News Dashboard Agent

Processes raw RSE (Research Software Engineering) articles to generate 2-3 sentence
summaries and assigns domain tags using advanced NLP techniques and AI skill orchestration.

Capabilities:
- NLP summarization using AI skill orchestrator
- Domain classification across 7 categories
- Key-phrase extraction
- Confidence scoring
- Integration with existing OpenAI and ML model registry

Author: RSE Summarization Engineer V1
Integration: AI News Dashboard NLP Pipeline
"""

import asyncio
import logging
import json
import re
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib
import aiohttp
import aiofiles
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DomainCategory(Enum):
    """RSE domain classification categories"""
    RESEARCH_COMPUTING = "research_computing"
    SOFTWARE_ENGINEERING = "software_engineering"
    DATA_SCIENCE = "data_science"
    HIGH_PERFORMANCE_COMPUTING = "high_performance_computing"
    MACHINE_LEARNING = "machine_learning"
    SCIENTIFIC_SOFTWARE = "scientific_software"
    DIGITAL_HUMANITIES = "digital_humanities"
    OTHER = "other"
    
    def __str__(self):
        return self.value

@dataclass
class RSEArticle:
    """Data structure for RSE article content"""
    title: str
    content: str
    url: Optional[str] = None
    date: Optional[str] = None
    source: Optional[str] = None
    author: Optional[str] = None
    raw_text: Optional[str] = None
    
    def __post_init__(self):
        if not self.raw_text:
            self.raw_text = f"{self.title}\n\n{self.content}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert article to dictionary"""
        return asdict(self)

@dataclass
class SummaryResult:
    """Result structure for summarization output"""
    title: str
    date: str
    summary: str
    domain: Union[str, DomainCategory]
    confidence_score: float
    key_phrases: List[str]
    processing_time: float
    model_used: str
    word_count: int
    
    def __post_init__(self):
        """Validate confidence score"""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {self.confidence_score}")
        
        # Convert domain to string if it's an enum
        if isinstance(self.domain, DomainCategory):
            self.domain = self.domain.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert result to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class ProcessingMetrics:
    """Metrics for tracking processing performance"""
    total_articles_processed: int = 0
    successful_summaries: int = 0
    failed_summaries: int = 0
    average_processing_time: float = 0.0
    domain_distribution: Dict[str, int] = None
    _processing_times: List[float] = None
    
    def __post_init__(self):
        if self.domain_distribution is None:
            self.domain_distribution = {}
        if self._processing_times is None:
            self._processing_times = []
    
    def record_success(self, processing_time: float):
        """Record a successful processing"""
        self.total_articles_processed += 1
        self.successful_summaries += 1
        self._processing_times.append(processing_time)
        self._update_average_time()
    
    def record_failure(self):
        """Record a failed processing"""
        self.total_articles_processed += 1
        self.failed_summaries += 1
    
    def _update_average_time(self):
        """Update average processing time"""
        if self._processing_times:
            self.average_processing_time = sum(self._processing_times) / len(self._processing_times)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_articles_processed == 0:
            return 0.0
        return self.successful_summaries / self.total_articles_processed

class RSESummarizationEngineer:
    """Main class for RSE article summarization and domain classification"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.cache_dir = Path(self.config.get('cache_dir', './cache/rse_summaries'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Domain classification keywords
        self.domain_keywords = {
            DomainCategory.RESEARCH_COMPUTING: [
                'research computing', 'computational research', 'scientific computing',
                'research infrastructure', 'cyberinfrastructure', 'grid computing'
            ],
            DomainCategory.SOFTWARE_ENGINEERING: [
                'software engineering', 'code quality', 'testing', 'debugging',
                'version control', 'continuous integration', 'agile', 'devops'
            ],
            DomainCategory.DATA_SCIENCE: [
                'data science', 'data analysis', 'statistics', 'visualization',
                'big data', 'analytics', 'data mining', 'data processing'
            ],
            DomainCategory.HIGH_PERFORMANCE_COMPUTING: [
                'hpc', 'high performance computing', 'parallel computing', 'supercomputing',
                'cluster computing', 'distributed computing', 'mpi', 'openmp'
            ],
            DomainCategory.MACHINE_LEARNING: [
                'machine learning', 'artificial intelligence', 'deep learning',
                'neural networks', 'ai', 'ml', 'tensorflow', 'pytorch'
            ],
            DomainCategory.SCIENTIFIC_SOFTWARE: [
                'scientific software', 'research software', 'computational tools',
                'simulation', 'modeling', 'numerical methods', 'algorithms'
            ],
            DomainCategory.DIGITAL_HUMANITIES: [
                'digital humanities', 'computational humanities', 'text mining',
                'digital scholarship', 'cultural analytics', 'humanities computing'
            ]
        }
        
        # Initialize metrics
        self.metrics = ProcessingMetrics()
        
        # API configurations
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = self.config.get('base_url', 'http://localhost:3000')
        
        logger.info(f"RSE Summarization Engineer initialized with config: {self.config.get('name', 'default')}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "name": "RSE Summarization Engineer",
            "version": "1.0.0",
            "cache_dir": "./cache/rse_summaries",
            "max_summary_length": 300,
            "min_summary_length": 100,
            "confidence_threshold": 0.7,
            "models": {
                "primary": "gpt-3.5-turbo",
                "fallback": "o4-mini-high"
            },
            "rate_limits": {
                "requests_per_minute": 60,
                "concurrent_requests": 5
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    async def _call_openai_api(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """Call OpenAI API for summarization"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        headers = {
            'Authorization': f'Bearer {self.openai_api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an expert RSE (Research Software Engineering) content summarizer. Generate concise, accurate summaries that capture the key technical and research aspects."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 256,
            "temperature": 0.3
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content'].strip()
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error {response.status}: {error_text}")
    
    async def _call_skill_orchestrator(self, article: RSEArticle) -> str:
        """Call the AI skill orchestrator for summarization"""
        try:
            # Try to call the local summarization API
            async with aiohttp.ClientSession() as session:
                payload = {
                    "text": article.raw_text,
                    "type": "brief",
                    "model": self.config["models"]["primary"]
                }
                
                async with session.post(
                    f"{self.base_url}/api/summarize",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('summary', '')
                    else:
                        logger.warning(f"Skill orchestrator failed with status {response.status}")
                        # Fallback to direct OpenAI call
                        return await self._fallback_summarization(article)
        except Exception as e:
            logger.warning(f"Skill orchestrator error: {e}")
            return await self._fallback_summarization(article)
    
    async def _fallback_summarization(self, article: RSEArticle) -> str:
        """Fallback summarization using direct API calls"""
        prompt = f"""
Summarize the following RSE article in 2-3 clear, concise sentences. Focus on:
1. The main technical contribution or research finding
2. The software engineering or computational aspect
3. The potential impact on the RSE community

Article:
Title: {article.title}
Content: {article.content[:2000]}...

Summary:"""
        
        try:
            return await self._call_openai_api(prompt)
        except Exception as e:
            logger.error(f"Fallback summarization failed: {e}")
            # Last resort: simple extractive summary
            return self._extractive_summary(article)
    
    def _extractive_summary(self, article: RSEArticle) -> str:
        """Simple extractive summarization as last resort"""
        sentences = re.split(r'[.!?]+', article.content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Take first sentence and one from middle
        if len(sentences) >= 2:
            summary_sentences = [sentences[0]]
            if len(sentences) > 2:
                summary_sentences.append(sentences[len(sentences)//2])
            return '. '.join(summary_sentences) + '.'
        elif sentences:
            return sentences[0] + '.'
        else:
            return f"Summary of {article.title}: Content analysis pending."
    
    def _classify_domain(self, article: RSEArticle) -> Tuple[str, float]:
        """Classify article domain using keyword matching and scoring"""
        text_lower = (article.title + " " + article.content).lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                # Count occurrences with different weights
                title_matches = article.title.lower().count(keyword.lower()) * 3
                content_matches = article.content.lower().count(keyword.lower())
                score += title_matches + content_matches
            
            domain_scores[domain] = score
        
        # Find best match
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            max_score = domain_scores[best_domain]
            
            # Calculate confidence (normalize by text length)
            text_length = len(text_lower.split())
            confidence = min(max_score / max(text_length * 0.1, 1), 1.0)
            
            if confidence >= self.config.get('confidence_threshold', 0.7):
                return best_domain.value, confidence
        
        return DomainCategory.OTHER.value, 0.5
    
    def _extract_key_phrases(self, article: RSEArticle, summary: str) -> List[str]:
        """Extract key phrases from article and summary"""
        # Combine title, summary, and content for phrase extraction
        combined_text = f"{article.title} {summary} {article.content}"
        
        # Simple keyword extraction based on frequency and domain relevance
        words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text.lower())
        word_freq = Counter(words)
        
        # Filter out common words and extract meaningful phrases
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use'}
        
        # Get technical terms and domain-specific keywords
        key_phrases = []
        for word, freq in word_freq.most_common(10):
            if word not in stop_words and len(word) > 3 and freq > 1:
                key_phrases.append(word)
        
        # Add domain-specific terms if found
        for domain_keywords in self.domain_keywords.values():
            for keyword in domain_keywords:
                if keyword.lower() in combined_text.lower() and keyword not in key_phrases:
                    key_phrases.append(keyword)
        
        return key_phrases[:8]  # Return top 8 key phrases
    
    def _generate_cache_key(self, article: RSEArticle) -> str:
        """Generate cache key for article"""
        content_hash = hashlib.md5(
            (article.title + article.content).encode('utf-8')
        ).hexdigest()
        return f"rse_summary_{content_hash}"
    
    async def _load_from_cache(self, cache_key: str) -> Optional[SummaryResult]:
        """Load summary from cache if available"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                async with aiofiles.open(cache_file, 'r') as f:
                    data = json.loads(await f.read())
                return SummaryResult(**data)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        
        return None
    
    async def _save_to_cache(self, cache_key: str, result: SummaryResult):
        """Save summary result to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(result.to_dict(), indent=2))
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    async def process_article(self, article: RSEArticle) -> SummaryResult:
        """Process a single RSE article to generate summary and classification"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(article)
        cached_result = await self._load_from_cache(cache_key)
        if cached_result:
            logger.info(f"Using cached result for article: {article.title[:50]}...")
            return cached_result
        
        try:
            # Step 1: Generate summary using skill orchestrator
            logger.info(f"Processing article: {article.title[:50]}...")
            summary = await self._call_skill_orchestrator(article)
            
            # Step 2: Classify domain
            domain, confidence = self._classify_domain(article)
            
            # Step 3: Extract key phrases
            key_phrases = self._extract_key_phrases(article, summary)
            
            # Step 4: Create result
            processing_time = time.time() - start_time
            result = SummaryResult(
                title=article.title,
                date=article.date or datetime.now().isoformat(),
                summary=summary,
                domain=domain,
                confidence_score=confidence,
                key_phrases=key_phrases,
                processing_time=processing_time,
                model_used=self.config["models"]["primary"],
                word_count=len(summary.split())
            )
            
            # Step 5: Cache result
            await self._save_to_cache(cache_key, result)
            
            # Update metrics
            self.metrics.record_success(processing_time)
            self.metrics.domain_distribution[domain] = self.metrics.domain_distribution.get(domain, 0) + 1
            
            logger.info(f"Successfully processed article: {article.title[:50]}... (domain: {domain}, confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process article {article.title[:50]}...: {e}")
            self.metrics.record_failure()
            
            # Return minimal result on failure
            return SummaryResult(
                title=article.title,
                date=article.date or datetime.now().isoformat(),
                summary=f"Processing failed: {str(e)[:100]}...",
                domain=DomainCategory.OTHER.value,
                confidence_score=0.0,
                key_phrases=[],
                processing_time=time.time() - start_time,
                model_used="error",
                word_count=0
            )
    
    async def process_batch(self, articles: List[RSEArticle]) -> List[SummaryResult]:
        """Process multiple articles in batch"""
        logger.info(f"Processing batch of {len(articles)} articles")
        
        # Update metrics will be handled by individual article processing
        
        # Process with concurrency limit
        semaphore = asyncio.Semaphore(self.config["rate_limits"]["concurrent_requests"])
        
        async def process_with_semaphore(article):
            async with semaphore:
                return await self.process_article(article)
        
        results = await asyncio.gather(
            *[process_with_semaphore(article) for article in articles],
            return_exceptions=True
        )
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception processing article {i}: {result}")
                self.metrics.record_failure()
            else:
                valid_results.append(result)
        
        # Update average processing time
        if valid_results:
            avg_time = sum(r.processing_time for r in valid_results) / len(valid_results)
            self.metrics.average_processing_time = avg_time
        
        logger.info(f"Batch processing complete: {len(valid_results)} successful, {len(results) - len(valid_results)} failed")
        return valid_results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current processing metrics"""
        return {
            "total_articles_processed": self.metrics.total_articles_processed,
            "successful_summaries": self.metrics.successful_summaries,
            "failed_summaries": self.metrics.failed_summaries,
            "success_rate": self.metrics.success_rate,
            "average_processing_time": self.metrics.average_processing_time,
            "domain_distribution": self.metrics.domain_distribution,
            "cache_size": len(list(self.cache_dir.glob("*.json"))),
            "config": self.config
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the summarization engine"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "config_loaded": self.config is not None
        }
        
        # Check OpenAI API key
        health_status["checks"]["openai_api_key"] = bool(self.openai_api_key)
        
        # Check cache directory
        health_status["checks"]["cache_directory"] = self.cache_dir.exists()
        
        # Simple connectivity check (non-async)
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            health_status["checks"]["skill_orchestrator"] = response.status_code == 200
        except:
            health_status["checks"]["skill_orchestrator"] = False
        
        # Overall status
        if not all(health_status["checks"].values()):
            health_status["status"] = "degraded"
        
        return health_status
    
    def _classify_domain(self, article: Union[RSEArticle, str]) -> Tuple[str, float]:
        """Classify article domain using keyword matching"""
        content = article.content if isinstance(article, RSEArticle) else article
        content_lower = content.lower()
        
        domain_scores = {}
        
        # Score each domain based on keyword matches
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    score += 1
            domain_scores[domain] = score / len(keywords) if keywords else 0
        
        # Find best match
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            confidence = domain_scores[best_domain]
            
            # Return string value of domain
            if confidence > 0:
                return best_domain.value, min(confidence, 1.0)
        
        return DomainCategory.OTHER.value, 0.1
    
    def _extract_key_phrases(self, article: Union[RSEArticle, str], summary: str = None) -> List[str]:
        """Extract key phrases from article content"""
        content = article.content if isinstance(article, RSEArticle) else article
        if summary:
            content += " " + summary
        
        # Simple keyword extraction using common RSE terms
        rse_terms = [
            'software engineering', 'research software', 'code quality',
            'testing', 'debugging', 'version control', 'continuous integration',
            'data science', 'machine learning', 'high performance computing',
            'scientific computing', 'digital humanities', 'algorithms',
            'programming', 'development', 'methodology', 'best practices'
        ]
        
        found_phrases = []
        content_lower = content.lower()
        
        for term in rse_terms:
            if term in content_lower:
                found_phrases.append(term)
        
        # Also extract capitalized words (likely important terms)
        import re
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', content)
        found_phrases.extend(capitalized_words[:5])  # Limit to 5
        
        return list(set(found_phrases))[:10]  # Return unique phrases, max 10
    
    def _generate_cache_key(self, article: Union[RSEArticle, str]) -> str:
        """Generate cache key for article"""
        if isinstance(article, RSEArticle):
            content = f"{article.title}|{article.content}"
        else:
            content = str(article)
        
        return hashlib.md5(content.encode()).hexdigest()

# CLI Interface
async def main():
    """CLI interface for RSE Summarization Engineer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RSE Summarization Engineer")
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--input', help='Input file with articles (JSON)')
    parser.add_argument('--output', help='Output file for results (JSON)')
    parser.add_argument('--health-check', action='store_true', help='Perform health check')
    parser.add_argument('--metrics', action='store_true', help='Show current metrics')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize engineer
    engineer = RSESummarizationEngineer(args.config)
    
    if args.health_check:
        health = await engineer.health_check()
        print(json.dumps(health, indent=2))
        return
    
    if args.metrics:
        metrics = engineer.get_metrics()
        print(json.dumps(metrics, indent=2))
        return
    
    if args.input:
        # Process articles from input file
        try:
            with open(args.input, 'r') as f:
                articles_data = json.load(f)
            
            articles = [RSEArticle(**article) for article in articles_data]
            results = await engineer.process_batch(articles)
            
            # Save results
            output_file = args.output or 'rse_summaries.json'
            with open(output_file, 'w') as f:
                json.dump([result.to_dict() for result in results], f, indent=2)
            
            print(f"Processed {len(results)} articles. Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to process input file: {e}")
    else:
        print("RSE Summarization Engineer ready. Use --input to process articles.")
        print("Example usage:")
        print("  python rse_summarization_engineer.py --input articles.json --output summaries.json")
        print("  python rse_summarization_engineer.py --health-check")
        print("  python rse_summarization_engineer.py --metrics")

if __name__ == "__main__":
    asyncio.run(main())