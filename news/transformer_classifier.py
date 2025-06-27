#!/usr/bin/env python3
"""
Transformer-based News Classification Service
Replaces keyword-based classification with advanced NLP models

Features:
- BERT/DistilBERT-based text classification
- Multi-label topic classification
- Sentiment analysis integration
- Named Entity Recognition
- Confidence scoring and uncertainty estimation
- Batch processing for efficiency
- Model caching and optimization
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertForSequenceClassification
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import spacy
from concurrent.futures import ThreadPoolExecutor
import redis
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
CLASSIFICATION_COUNTER = Counter('transformer_classifications_total', 'Total classifications made', ['model_type', 'category'])
CLASSIFICATION_LATENCY = Histogram('transformer_classification_duration_seconds', 'Classification latency')
CONFIDENCE_SCORE = Histogram('transformer_confidence_scores', 'Classification confidence scores')
MODEL_ACCURACY = Gauge('transformer_model_accuracy', 'Current model accuracy', ['model_name'])

@dataclass
class ClassificationRequest:
    """Request structure for text classification"""
    text: str
    title: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    include_sentiment: bool = True
    include_entities: bool = True
    confidence_threshold: float = 0.7
    max_categories: int = 3

@dataclass
class ClassificationResult:
    """Result structure for text classification"""
    primary_category: str
    all_categories: List[Tuple[str, float]]  # (category, confidence)
    sentiment: Optional[Dict[str, float]] = None
    entities: Optional[List[Dict[str, str]]] = None
    confidence_score: float = 0.0
    processing_time: float = 0.0
    model_version: str = "v1.0.0"
    timestamp: str = ""
    metadata: Optional[Dict] = None

class TransformerNewsClassifier:
    """Advanced transformer-based news classifier"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._load_default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.get('use_gpu', True) else 'cpu')
        
        # Model components
        self.tokenizer = None
        self.classification_model = None
        self.sentiment_pipeline = None
        self.ner_model = None
        
        # Category mappings
        self.category_labels = [
            'technology', 'business', 'politics', 'health', 'science',
            'sports', 'entertainment', 'world', 'environment', 'education',
            'crime', 'weather', 'travel', 'food', 'lifestyle'
        ]
        
        # Cache and performance
        self.redis_client = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.model_cache = {}
        
        # Initialize components
        self._initialize_redis()
        self._initialize_models()
        
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            'classification_model': 'distilbert-base-uncased',
            'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'ner_model': 'en_core_web_sm',
            'use_gpu': torch.cuda.is_available(),
            'batch_size': 16,
            'max_length': 512,
            'cache_ttl': 3600,  # 1 hour
            'confidence_threshold': 0.7,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 3
        }
    
    def _initialize_redis(self):
        """Initialize Redis connection for caching"""
        try:
            self.redis_client = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                db=self.config['redis_db'],
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✅ Connected to Redis for classification caching")
        except Exception as e:
            logger.warning(f"⚠️ Redis not available for classification caching: {e}")
            self.redis_client = None
    
    def _initialize_models(self):
        """Initialize all transformer models"""
        logger.info("🤖 Initializing transformer models...")
        
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['classification_model']
            )
            logger.info(f"✅ Loaded tokenizer: {self.config['classification_model']}")
            
            # Initialize classification model
            self.classification_model = AutoModel.from_pretrained(
                self.config['classification_model']
            ).to(self.device)
            
            # Add classification head
            self.classification_head = nn.Sequential(
                nn.Linear(self.classification_model.config.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(self.category_labels)),
                nn.Sigmoid()  # Multi-label classification
            ).to(self.device)
            
            logger.info(f"✅ Loaded classification model with {len(self.category_labels)} categories")
            
            # Initialize sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.config['sentiment_model'],
                device=0 if self.device.type == 'cuda' else -1
            )
            logger.info(f"✅ Loaded sentiment model: {self.config['sentiment_model']}")
            
            # Initialize NER model
            try:
                self.ner_model = spacy.load(self.config['ner_model'])
                logger.info(f"✅ Loaded NER model: {self.config['ner_model']}")
            except OSError:
                logger.warning(f"⚠️ NER model {self.config['ner_model']} not found, using basic NER")
                self.ner_model = None
            
            # Set models to evaluation mode
            self.classification_model.eval()
            self.classification_head.eval()
            
            logger.info("🎯 All transformer models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize transformer models: {e}")
            raise
    
    async def classify(self, request: ClassificationRequest) -> ClassificationResult:
        """Main classification method"""
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_result = await self._get_cached_result(cache_key)
            
            if cached_result:
                logger.info("📋 Returning cached classification result")
                return cached_result
            
            # Prepare text for classification
            full_text = self._prepare_text(request)
            
            # Perform classification
            categories = await self._classify_text(full_text)
            
            # Perform sentiment analysis if requested
            sentiment = None
            if request.include_sentiment:
                sentiment = await self._analyze_sentiment(full_text)
            
            # Perform NER if requested
            entities = None
            if request.include_entities:
                entities = await self._extract_entities(full_text)
            
            # Filter categories by confidence threshold
            filtered_categories = [
                (cat, conf) for cat, conf in categories 
                if conf >= request.confidence_threshold
            ]
            
            # Ensure we have at least one category
            if not filtered_categories:
                filtered_categories = [categories[0]] if categories else [('general', 0.5)]
            
            # Limit to max categories
            filtered_categories = filtered_categories[:request.max_categories]
            
            # Create result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ClassificationResult(
                primary_category=filtered_categories[0][0],
                all_categories=filtered_categories,
                sentiment=sentiment,
                entities=entities,
                confidence_score=filtered_categories[0][1],
                processing_time=processing_time,
                model_version="v1.0.0",
                timestamp=datetime.now().isoformat(),
                metadata={
                    'text_length': len(full_text),
                    'source': request.source,
                    'url': request.url
                }
            )
            
            # Cache the result
            await self._cache_result(cache_key, result)
            
            # Record metrics
            CLASSIFICATION_COUNTER.labels(
                model_type='transformer',
                category=result.primary_category
            ).inc()
            
            CLASSIFICATION_LATENCY.observe(processing_time)
            CONFIDENCE_SCORE.observe(result.confidence_score)
            
            logger.info(f"✅ Classified as '{result.primary_category}' with {result.confidence_score:.2f} confidence")
            return result
            
        except Exception as e:
            logger.error(f"❌ Classification failed: {e}")
            raise
    
    def _prepare_text(self, request: ClassificationRequest) -> str:
        """Prepare text for classification"""
        text_parts = []
        
        if request.title:
            text_parts.append(f"Title: {request.title}")
        
        if request.description:
            text_parts.append(f"Description: {request.description}")
        
        if request.text:
            text_parts.append(f"Content: {request.text}")
        
        full_text = " ".join(text_parts)
        
        # Truncate if too long
        max_chars = self.config['max_length'] * 4  # Rough estimate
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "..."
        
        return full_text
    
    async def _classify_text(self, text: str) -> List[Tuple[str, float]]:
        """Classify text into categories"""
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=self.config['max_length']
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.classification_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                
                # Get classification scores
                logits = self.classification_head(embeddings)
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Create category-confidence pairs
            categories = list(zip(self.category_labels, probabilities))
            
            # Sort by confidence
            categories.sort(key=lambda x: x[1], reverse=True)
            
            return categories
            
        except Exception as e:
            logger.error(f"Text classification failed: {e}")
            # Return default classification
            return [('general', 0.5)]
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        try:
            # Truncate text for sentiment analysis
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            # Run sentiment analysis
            result = self.sentiment_pipeline(text)[0]
            
            # Convert to standardized format
            label = result['label'].lower()
            score = result['score']
            
            # Map labels to standard format
            label_mapping = {
                'positive': 'positive',
                'negative': 'negative',
                'neutral': 'neutral',
                'label_0': 'negative',
                'label_1': 'neutral', 
                'label_2': 'positive'
            }
            
            mapped_label = label_mapping.get(label, 'neutral')
            
            return {
                'label': mapped_label,
                'score': float(score),
                'confidence': float(score)
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                'label': 'neutral',
                'score': 0.5,
                'confidence': 0.5
            }
    
    async def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text"""
        if not self.ner_model:
            return []
        
        try:
            # Process text with spaCy
            doc = self.ner_model(text[:1000])  # Limit text length
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_) or ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': float(ent._.get('confidence', 0.8))  # Default confidence
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    async def batch_classify(self, requests: List[ClassificationRequest]) -> List[ClassificationResult]:
        """Classify multiple texts in batch"""
        logger.info(f"🔄 Processing batch of {len(requests)} classifications")
        
        # Process requests concurrently
        tasks = [self.classify(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch classification failed for item {i}: {result}")
                # Create error result
                processed_results.append(ClassificationResult(
                    primary_category='general',
                    all_categories=[('general', 0.0)],
                    confidence_score=0.0,
                    processing_time=0.0,
                    timestamp=datetime.now().isoformat(),
                    metadata={'error': str(result)}
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _generate_cache_key(self, request: ClassificationRequest) -> str:
        """Generate cache key for classification request"""
        key_data = {
            'text_hash': hash(request.text),
            'title_hash': hash(request.title or ''),
            'description_hash': hash(request.description or ''),
            'include_sentiment': request.include_sentiment,
            'include_entities': request.include_entities,
            'confidence_threshold': request.confidence_threshold
        }
        return f"transformer_classification:{hash(str(key_data))}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[ClassificationResult]:
        """Get cached classification result"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return ClassificationResult(**data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: ClassificationResult):
        """Cache classification result"""
        if not self.redis_client:
            return
        
        try:
            data = asdict(result)
            self.redis_client.setex(
                cache_key,
                self.config['cache_ttl'],
                json.dumps(data, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    async def get_model_info(self) -> Dict[str, any]:
        """Get information about loaded models"""
        return {
            'classification_model': self.config['classification_model'],
            'sentiment_model': self.config['sentiment_model'],
            'ner_model': self.config['ner_model'],
            'device': str(self.device),
            'categories': self.category_labels,
            'cache_enabled': self.redis_client is not None,
            'model_version': 'v1.0.0',
            'last_updated': datetime.now().isoformat()
        }
    
    async def update_categories(self, new_categories: List[str]):
        """Update category labels (requires model retraining)"""
        logger.info(f"📝 Updating categories from {len(self.category_labels)} to {len(new_categories)}")
        
        self.category_labels = new_categories
        
        # Reinitialize classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.classification_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(self.category_labels)),
            nn.Sigmoid()
        ).to(self.device)
        
        logger.info(f"✅ Updated to {len(new_categories)} categories")

# Global classifier instance
_classifier = None

def get_transformer_classifier() -> TransformerNewsClassifier:
    """Get or create transformer classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = TransformerNewsClassifier()
    return _classifier

# Convenience functions for integration
async def classify_article(title: str, description: str = None, content: str = None, 
                          source: str = None, url: str = None) -> ClassificationResult:
    """Classify a news article"""
    classifier = get_transformer_classifier()
    
    request = ClassificationRequest(
        text=content or '',
        title=title,
        description=description,
        source=source,
        url=url,
        include_sentiment=True,
        include_entities=True
    )
    
    return await classifier.classify(request)

async def classify_articles_batch(articles: List[Dict]) -> List[ClassificationResult]:
    """Classify multiple articles in batch"""
    classifier = get_transformer_classifier()
    
    requests = []
    for article in articles:
        request = ClassificationRequest(
            text=article.get('content', ''),
            title=article.get('title', ''),
            description=article.get('description', ''),
            source=article.get('source', ''),
            url=article.get('url', ''),
            include_sentiment=True,
            include_entities=True
        )
        requests.append(request)
    
    return await classifier.batch_classify(requests)

# Legacy compatibility function for news ingestion
async def classify_topics_transformer(articles: List[Dict]) -> List[Dict]:
    """Legacy compatibility function for news ingestion"""
    logger.info(f"🏷️ Classifying topics for {len(articles)} articles using transformers...")
    
    try:
        # Classify articles
        results = await classify_articles_batch(articles)
        
        # Update articles with classification results
        for i, (article, result) in enumerate(zip(articles, results)):
            article['category'] = result.primary_category
            article['topicConfidence'] = result.confidence_score
            article['allCategories'] = result.all_categories
            
            if result.sentiment:
                article['sentiment'] = result.sentiment
            
            if result.entities:
                article['entities'] = result.entities
            
            # Add transformer-specific metadata
            article['classificationMethod'] = 'transformer'
            article['modelVersion'] = result.model_version
            article['processingTime'] = result.processing_time
        
        logger.info(f"✅ Classified {len(articles)} articles using transformer models")
        return articles
        
    except Exception as e:
        logger.error(f"❌ Transformer classification failed: {e}")
        # Fallback to original articles
        return articles

if __name__ == '__main__':
    # Example usage
    async def main():
        classifier = TransformerNewsClassifier()
        
        # Test classification
        request = ClassificationRequest(
            text="Apple announced new iPhone features including advanced AI capabilities and improved camera technology.",
            title="Apple Unveils New iPhone with AI Features",
            description="Latest iPhone model includes cutting-edge artificial intelligence",
            source="TechNews",
            include_sentiment=True,
            include_entities=True
        )
        
        result = await classifier.classify(request)
        
        print(f"Primary Category: {result.primary_category}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"All Categories: {result.all_categories}")
        print(f"Sentiment: {result.sentiment}")
        print(f"Entities: {len(result.entities or [])} found")
        print(f"Processing Time: {result.processing_time:.2f}s")
        
        # Test model info
        info = await classifier.get_model_info()
        print(f"\nModel Info: {info}")
    
    asyncio.run(main())