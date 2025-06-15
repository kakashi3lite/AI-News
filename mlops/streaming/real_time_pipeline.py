#!/usr/bin/env python3
"""
Dr. NewsForge's Advanced Real-Time News Processing Pipeline

Implements high-throughput streaming analytics for real-time news processing,
event detection, and intelligent content routing.

Features:
- Apache Kafka integration for event streaming
- Real-time sentiment analysis and trend detection
- Event-driven architecture with microservices
- Stream processing with Apache Flink/Spark Streaming
- Real-time dashboards and alerting
- Geospatial news analysis
- Multi-language content processing
- Anomaly detection in news patterns

Author: Dr. Nova "NewsForge" Arclight
Version: 2.0.0
"""

import os
import json
import time
import logging
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import uuid
import hashlib
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertModel
)
from sentence_transformers import SentenceTransformer
import spacy
from textblob import TextBlob
import langdetect
from googletrans import Translator

import kafka
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import redis
import elasticsearch
from elasticsearch import Elasticsearch
import pymongo
from pymongo import MongoClient
import psycopg2
from sqlalchemy import create_engine, text

import requests
from flask import Flask, request, jsonify, Response
from flask_socketio import SocketIO, emit
import websocket
from celery import Celery
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import mlflow
import wandb

import geopandas as gpd
from shapely.geometry import Point
import folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
NEWS_PROCESSED = Counter('news_articles_processed_total', 'Total news articles processed', ['source', 'language'])
PROCESSING_LATENCY = Histogram('news_processing_latency_seconds', 'News processing latency', ['stage'])
SENTIMENT_DISTRIBUTION = Counter('sentiment_distribution_total', 'Sentiment distribution', ['sentiment'])
TREND_ALERTS = Counter('trend_alerts_total', 'Trend alerts generated', ['trend_type'])
GEO_EVENTS = Counter('geo_events_total', 'Geographical events detected', ['country', 'event_type'])
STREAM_THROUGHPUT = Gauge('stream_throughput_articles_per_second', 'Stream processing throughput')
ANOMALY_SCORE = Gauge('anomaly_score', 'Current anomaly score', ['detector_type'])

@dataclass
class NewsEvent:
    """Represents a news event in the streaming pipeline."""
    event_id: str
    timestamp: datetime
    source: str
    title: str
    content: str
    url: str
    language: str
    sentiment: Optional[float] = None
    entities: Optional[List[Dict[str, Any]]] = None
    location: Optional[Tuple[float, float]] = None
    category: Optional[str] = None
    keywords: Optional[List[str]] = None
    embedding: Optional[List[float]] = None
    confidence: Optional[float] = None
    processed_at: Optional[datetime] = None

@dataclass
class TrendAlert:
    """Represents a detected trend or anomaly."""
    alert_id: str
    trend_type: str  # 'emerging_topic', 'sentiment_shift', 'geo_cluster', 'anomaly'
    severity: str    # 'low', 'medium', 'high', 'critical'
    description: str
    affected_articles: List[str]
    confidence: float
    detected_at: datetime
    location: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class StreamingConfig:
    """Configuration for streaming pipeline."""
    kafka_bootstrap_servers: List[str]
    kafka_topics: Dict[str, str]
    redis_host: str
    redis_port: int
    elasticsearch_host: str
    elasticsearch_port: int
    mongodb_uri: str
    postgres_uri: str
    batch_size: int = 100
    processing_timeout: int = 30
    max_workers: int = 10
    enable_gpu: bool = True
    languages: List[str] = None

class NewsPreprocessor:
    """Advanced news preprocessing with multi-language support."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        
        # Language detection and translation
        self.translator = Translator()
        
        # NLP models
        self.nlp_models = {}
        self.load_nlp_models()
        
        # Sentiment analysis
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if config.enable_gpu and torch.cuda.is_available() else -1
        )
        
        # Named Entity Recognition
        self.ner_pipeline = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple",
            device=0 if config.enable_gpu and torch.cuda.is_available() else -1
        )
        
        # Embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Geocoder
        self.geocoder = Nominatim(user_agent="news_pipeline")
        
        logger.info("News preprocessor initialized")
    
    def load_nlp_models(self):
        """Load language-specific NLP models."""
        languages = self.config.languages or ['en', 'es', 'fr', 'de', 'zh']
        
        for lang in languages:
            try:
                if lang == 'en':
                    self.nlp_models[lang] = spacy.load('en_core_web_sm')
                elif lang == 'es':
                    self.nlp_models[lang] = spacy.load('es_core_news_sm')
                elif lang == 'fr':
                    self.nlp_models[lang] = spacy.load('fr_core_news_sm')
                elif lang == 'de':
                    self.nlp_models[lang] = spacy.load('de_core_news_sm')
                elif lang == 'zh':
                    self.nlp_models[lang] = spacy.load('zh_core_web_sm')
                else:
                    # Fallback to English
                    self.nlp_models[lang] = spacy.load('en_core_web_sm')
            except OSError:
                logger.warning(f"SpaCy model for {lang} not found, using English")
                self.nlp_models[lang] = spacy.load('en_core_web_sm')
    
    async def process_news_event(self, raw_data: Dict[str, Any]) -> NewsEvent:
        """Process raw news data into structured NewsEvent."""
        start_time = time.time()
        
        # Create base event
        event = NewsEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.fromisoformat(raw_data.get('timestamp', datetime.now().isoformat())),
            source=raw_data.get('source', 'unknown'),
            title=raw_data.get('title', ''),
            content=raw_data.get('content', ''),
            url=raw_data.get('url', ''),
            language=raw_data.get('language', 'unknown')
        )
        
        # Detect language if not provided
        if event.language == 'unknown':
            event.language = self.detect_language(event.title + ' ' + event.content)
        
        # Process content
        await self.extract_entities(event)
        await self.analyze_sentiment(event)
        await self.extract_keywords(event)
        await self.generate_embedding(event)
        await self.extract_location(event)
        await self.classify_category(event)
        
        event.processed_at = datetime.now()
        
        # Record metrics
        processing_time = time.time() - start_time
        PROCESSING_LATENCY.labels(stage='preprocessing').observe(processing_time)
        NEWS_PROCESSED.labels(source=event.source, language=event.language).inc()
        
        return event
    
    def detect_language(self, text: str) -> str:
        """Detect language of text."""
        try:
            return langdetect.detect(text)
        except:
            return 'en'  # Default to English
    
    async def extract_entities(self, event: NewsEvent):
        """Extract named entities from news content."""
        try:
            # Use transformer-based NER
            text = f"{event.title} {event.content}"
            entities = self.ner_pipeline(text)
            
            # Process entities
            processed_entities = []
            for entity in entities:
                processed_entities.append({
                    'text': entity['word'],
                    'label': entity['entity_group'],
                    'confidence': entity['score'],
                    'start': entity.get('start', 0),
                    'end': entity.get('end', 0)
                })
            
            event.entities = processed_entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            event.entities = []
    
    async def analyze_sentiment(self, event: NewsEvent):
        """Analyze sentiment of news content."""
        try:
            text = f"{event.title} {event.content}"
            
            # Use transformer-based sentiment analysis
            result = self.sentiment_analyzer(text[:512])  # Limit text length
            
            # Convert to numerical score
            if result[0]['label'] == 'POSITIVE':
                event.sentiment = result[0]['score']
            elif result[0]['label'] == 'NEGATIVE':
                event.sentiment = -result[0]['score']
            else:  # NEUTRAL
                event.sentiment = 0.0
            
            # Record sentiment distribution
            sentiment_label = 'positive' if event.sentiment > 0.1 else 'negative' if event.sentiment < -0.1 else 'neutral'
            SENTIMENT_DISTRIBUTION.labels(sentiment=sentiment_label).inc()
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            event.sentiment = 0.0
    
    async def extract_keywords(self, event: NewsEvent):
        """Extract keywords from news content."""
        try:
            # Use spaCy for keyword extraction
            nlp = self.nlp_models.get(event.language, self.nlp_models['en'])
            doc = nlp(event.content)
            
            # Extract keywords based on POS tags and entities
            keywords = set()
            
            # Add named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']:
                    keywords.add(ent.text.lower())
            
            # Add important nouns and adjectives
            for token in doc:
                if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                    not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2):
                    keywords.add(token.lemma_.lower())
            
            event.keywords = list(keywords)[:20]  # Limit to top 20 keywords
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            event.keywords = []
    
    async def generate_embedding(self, event: NewsEvent):
        """Generate semantic embedding for news content."""
        try:
            text = f"{event.title} {event.content}"
            embedding = self.embedding_model.encode(text[:512])
            event.embedding = embedding.tolist()
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            event.embedding = []
    
    async def extract_location(self, event: NewsEvent):
        """Extract geographical location from news content."""
        try:
            # Look for location entities
            location_entities = []
            if event.entities:
                location_entities = [
                    ent['text'] for ent in event.entities 
                    if ent['label'] in ['LOC', 'GPE']
                ]
            
            # Try to geocode the first location found
            if location_entities:
                location_name = location_entities[0]
                location = self.geocoder.geocode(location_name)
                
                if location:
                    event.location = (location.latitude, location.longitude)
            
        except Exception as e:
            logger.error(f"Location extraction failed: {e}")
            event.location = None
    
    async def classify_category(self, event: NewsEvent):
        """Classify news into categories."""
        try:
            # Simple keyword-based classification
            text = f"{event.title} {event.content}".lower()
            
            categories = {
                'politics': ['election', 'government', 'policy', 'politician', 'vote', 'congress', 'senate'],
                'business': ['economy', 'market', 'stock', 'company', 'finance', 'trade', 'investment'],
                'technology': ['tech', 'ai', 'software', 'computer', 'internet', 'digital', 'innovation'],
                'health': ['health', 'medical', 'hospital', 'doctor', 'disease', 'treatment', 'vaccine'],
                'sports': ['sport', 'game', 'team', 'player', 'match', 'championship', 'olympic'],
                'entertainment': ['movie', 'music', 'celebrity', 'film', 'actor', 'entertainment', 'show']
            }
            
            category_scores = {}
            for category, keywords in categories.items():
                score = sum(1 for keyword in keywords if keyword in text)
                category_scores[category] = score
            
            # Assign category with highest score
            if category_scores:
                event.category = max(category_scores, key=category_scores.get)
                if category_scores[event.category] == 0:
                    event.category = 'general'
            else:
                event.category = 'general'
            
        except Exception as e:
            logger.error(f"Category classification failed: {e}")
            event.category = 'general'

class StreamProcessor:
    """Real-time stream processing engine."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        
        # Kafka setup
        self.producer = KafkaProducer(
            bootstrap_servers=config.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        
        self.consumer = KafkaConsumer(
            config.kafka_topics.get('raw_news', 'raw_news'),
            bootstrap_servers=config.kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='news_processor',
            auto_offset_reset='latest'
        )
        
        # Storage backends
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            decode_responses=True
        )
        
        self.es_client = Elasticsearch([{
            'host': config.elasticsearch_host,
            'port': config.elasticsearch_port
        }])
        
        self.mongo_client = MongoClient(config.mongodb_uri)
        self.mongo_db = self.mongo_client.news_pipeline
        
        # Components
        self.preprocessor = NewsPreprocessor(config)
        self.trend_detector = TrendDetector(config)
        self.anomaly_detector = AnomalyDetector(config)
        
        # Processing state
        self.processing_stats = {
            'total_processed': 0,
            'processing_rate': 0.0,
            'last_update': time.time()
        }
        
        # Background tasks
        self.running = True
        self.worker_threads = []
        
        logger.info("Stream processor initialized")
    
    def start(self):
        """Start the stream processing pipeline."""
        logger.info("Starting stream processor")
        
        # Start worker threads
        for i in range(self.config.max_workers):
            worker = threading.Thread(
                target=self._process_stream,
                name=f"StreamWorker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitor_performance,
            daemon=True
        )
        monitor_thread.start()
        
        logger.info(f"Started {len(self.worker_threads)} stream workers")
    
    def _process_stream(self):
        """Process incoming news stream."""
        while self.running:
            try:
                # Get batch of messages
                message_batch = []
                start_time = time.time()
                
                for message in self.consumer:
                    message_batch.append(message.value)
                    
                    if (len(message_batch) >= self.config.batch_size or 
                        time.time() - start_time > self.config.processing_timeout):
                        break
                
                if message_batch:
                    asyncio.run(self._process_batch(message_batch))
                
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
                time.sleep(5)
    
    async def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of news messages."""
        start_time = time.time()
        
        # Process messages concurrently
        tasks = []
        for raw_data in batch:
            task = asyncio.create_task(self._process_single_message(raw_data))
            tasks.append(task)
        
        # Wait for all tasks to complete
        processed_events = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        valid_events = [
            event for event in processed_events 
            if isinstance(event, NewsEvent)
        ]
        
        # Store processed events
        if valid_events:
            await self._store_events(valid_events)
            await self._detect_trends(valid_events)
            await self._detect_anomalies(valid_events)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.processing_stats['total_processed'] += len(valid_events)
        
        # Update throughput metric
        if processing_time > 0:
            throughput = len(valid_events) / processing_time
            STREAM_THROUGHPUT.set(throughput)
        
        logger.info(f"Processed batch of {len(valid_events)} events in {processing_time:.2f}s")
    
    async def _process_single_message(self, raw_data: Dict[str, Any]) -> NewsEvent:
        """Process a single news message."""
        try:
            # Preprocess the news event
            event = await self.preprocessor.process_news_event(raw_data)
            return event
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            raise
    
    async def _store_events(self, events: List[NewsEvent]):
        """Store processed events in various backends."""
        try:
            # Store in Elasticsearch for search
            es_docs = []
            for event in events:
                doc = {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp,
                    'source': event.source,
                    'title': event.title,
                    'content': event.content,
                    'url': event.url,
                    'language': event.language,
                    'sentiment': event.sentiment,
                    'entities': event.entities,
                    'location': event.location,
                    'category': event.category,
                    'keywords': event.keywords,
                    'processed_at': event.processed_at
                }
                es_docs.append({
                    '_index': f"news-{datetime.now().strftime('%Y-%m')}",
                    '_id': event.event_id,
                    '_source': doc
                })
            
            # Bulk insert to Elasticsearch
            if es_docs:
                elasticsearch.helpers.bulk(self.es_client, es_docs)
            
            # Store in MongoDB for analytics
            mongo_docs = [asdict(event) for event in events]
            self.mongo_db.news_events.insert_many(mongo_docs)
            
            # Cache recent events in Redis
            for event in events:
                self.redis_client.zadd(
                    'recent_news',
                    {event.event_id: event.timestamp.timestamp()}
                )
                self.redis_client.setex(
                    f"news:{event.event_id}",
                    3600,  # 1 hour TTL
                    json.dumps(asdict(event), default=str)
                )
            
            # Trim Redis cache to keep only recent items
            cutoff_time = (datetime.now() - timedelta(hours=24)).timestamp()
            self.redis_client.zremrangebyscore('recent_news', 0, cutoff_time)
            
        except Exception as e:
            logger.error(f"Failed to store events: {e}")
    
    async def _detect_trends(self, events: List[NewsEvent]):
        """Detect trends in the news events."""
        try:
            trends = await self.trend_detector.detect_trends(events)
            
            for trend in trends:
                # Publish trend alert
                self.producer.send(
                    self.config.kafka_topics.get('trend_alerts', 'trend_alerts'),
                    key=trend.alert_id,
                    value=asdict(trend)
                )
                
                # Record metric
                TREND_ALERTS.labels(trend_type=trend.trend_type).inc()
                
                logger.info(f"Detected trend: {trend.description}")
        
        except Exception as e:
            logger.error(f"Trend detection failed: {e}")
    
    async def _detect_anomalies(self, events: List[NewsEvent]):
        """Detect anomalies in the news stream."""
        try:
            anomalies = await self.anomaly_detector.detect_anomalies(events)
            
            for anomaly in anomalies:
                # Publish anomaly alert
                self.producer.send(
                    self.config.kafka_topics.get('anomaly_alerts', 'anomaly_alerts'),
                    key=anomaly.alert_id,
                    value=asdict(anomaly)
                )
                
                # Update anomaly score metric
                ANOMALY_SCORE.labels(detector_type=anomaly.trend_type).set(anomaly.confidence)
                
                logger.warning(f"Detected anomaly: {anomaly.description}")
        
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
    
    def _monitor_performance(self):
        """Monitor processing performance."""
        while self.running:
            try:
                current_time = time.time()
                time_diff = current_time - self.processing_stats['last_update']
                
                if time_diff >= 60:  # Update every minute
                    # Calculate processing rate
                    rate = self.processing_stats['total_processed'] / time_diff
                    self.processing_stats['processing_rate'] = rate
                    self.processing_stats['last_update'] = current_time
                    self.processing_stats['total_processed'] = 0
                    
                    logger.info(f"Processing rate: {rate:.2f} events/second")
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(60)
    
    def stop(self):
        """Stop the stream processor."""
        logger.info("Stopping stream processor")
        self.running = False
        
        # Close connections
        self.producer.close()
        self.consumer.close()
        self.redis_client.close()
        self.es_client.close()
        self.mongo_client.close()

class TrendDetector:
    """Advanced trend detection system."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        
        # Trend detection models
        self.topic_model = None
        self.clustering_model = DBSCAN(eps=0.3, min_samples=5)
        
        # Historical data for trend analysis
        self.keyword_history = defaultdict(list)
        self.sentiment_history = defaultdict(list)
        self.location_history = defaultdict(list)
        
        # Trend thresholds
        self.trend_thresholds = {
            'keyword_spike': 3.0,      # 3x normal frequency
            'sentiment_shift': 0.3,     # 30% sentiment change
            'geo_cluster': 5,           # 5+ events in same location
            'topic_emergence': 0.8      # 80% confidence for new topic
        }
        
        logger.info("Trend detector initialized")
    
    async def detect_trends(self, events: List[NewsEvent]) -> List[TrendAlert]:
        """Detect various types of trends in news events."""
        trends = []
        
        # Detect keyword spikes
        keyword_trends = await self._detect_keyword_spikes(events)
        trends.extend(keyword_trends)
        
        # Detect sentiment shifts
        sentiment_trends = await self._detect_sentiment_shifts(events)
        trends.extend(sentiment_trends)
        
        # Detect geographical clusters
        geo_trends = await self._detect_geo_clusters(events)
        trends.extend(geo_trends)
        
        # Detect emerging topics
        topic_trends = await self._detect_emerging_topics(events)
        trends.extend(topic_trends)
        
        return trends
    
    async def _detect_keyword_spikes(self, events: List[NewsEvent]) -> List[TrendAlert]:
        """Detect sudden spikes in keyword frequency."""
        trends = []
        
        try:
            # Count keyword frequencies
            current_keywords = Counter()
            for event in events:
                if event.keywords:
                    current_keywords.update(event.keywords)
            
            # Compare with historical averages
            for keyword, count in current_keywords.items():
                historical_counts = self.keyword_history[keyword]
                
                if len(historical_counts) >= 10:  # Need sufficient history
                    avg_count = np.mean(historical_counts)
                    std_count = np.std(historical_counts)
                    
                    # Detect spike
                    if count > avg_count + (self.trend_thresholds['keyword_spike'] * std_count):
                        affected_articles = [
                            event.event_id for event in events 
                            if event.keywords and keyword in event.keywords
                        ]
                        
                        trend = TrendAlert(
                            alert_id=str(uuid.uuid4()),
                            trend_type='keyword_spike',
                            severity='medium' if count < avg_count * 5 else 'high',
                            description=f"Spike detected in keyword '{keyword}': {count} vs avg {avg_count:.1f}",
                            affected_articles=affected_articles,
                            confidence=min(1.0, (count - avg_count) / (avg_count + 1)),
                            detected_at=datetime.now(),
                            metadata={'keyword': keyword, 'count': count, 'avg_count': avg_count}
                        )
                        trends.append(trend)
                
                # Update history
                self.keyword_history[keyword].append(count)
                if len(self.keyword_history[keyword]) > 100:  # Keep last 100 data points
                    self.keyword_history[keyword].pop(0)
        
        except Exception as e:
            logger.error(f"Keyword spike detection failed: {e}")
        
        return trends
    
    async def _detect_sentiment_shifts(self, events: List[NewsEvent]) -> List[TrendAlert]:
        """Detect significant sentiment shifts."""
        trends = []
        
        try:
            # Group events by category
            category_sentiments = defaultdict(list)
            for event in events:
                if event.sentiment is not None and event.category:
                    category_sentiments[event.category].append(event.sentiment)
            
            # Analyze sentiment shifts
            for category, sentiments in category_sentiments.items():
                if len(sentiments) >= 5:  # Need sufficient data
                    current_avg = np.mean(sentiments)
                    
                    historical_sentiments = self.sentiment_history[category]
                    if len(historical_sentiments) >= 10:
                        historical_avg = np.mean(historical_sentiments[-10:])  # Last 10 periods
                        
                        # Detect significant shift
                        shift = abs(current_avg - historical_avg)
                        if shift > self.trend_thresholds['sentiment_shift']:
                            affected_articles = [
                                event.event_id for event in events 
                                if event.category == category
                            ]
                            
                            direction = 'positive' if current_avg > historical_avg else 'negative'
                            
                            trend = TrendAlert(
                                alert_id=str(uuid.uuid4()),
                                trend_type='sentiment_shift',
                                severity='medium' if shift < 0.5 else 'high',
                                description=f"Sentiment shift in {category}: {direction} change of {shift:.2f}",
                                affected_articles=affected_articles,
                                confidence=min(1.0, shift / 0.5),
                                detected_at=datetime.now(),
                                metadata={
                                    'category': category,
                                    'current_sentiment': current_avg,
                                    'historical_sentiment': historical_avg,
                                    'shift': shift
                                }
                            )
                            trends.append(trend)
                    
                    # Update history
                    self.sentiment_history[category].append(current_avg)
                    if len(self.sentiment_history[category]) > 50:
                        self.sentiment_history[category].pop(0)
        
        except Exception as e:
            logger.error(f"Sentiment shift detection failed: {e}")
        
        return trends
    
    async def _detect_geo_clusters(self, events: List[NewsEvent]) -> List[TrendAlert]:
        """Detect geographical clusters of news events."""
        trends = []
        
        try:
            # Filter events with location data
            geo_events = [event for event in events if event.location]
            
            if len(geo_events) >= self.trend_thresholds['geo_cluster']:
                # Extract coordinates
                coordinates = np.array([event.location for event in geo_events])
                
                # Perform clustering
                clusters = self.clustering_model.fit_predict(coordinates)
                
                # Analyze clusters
                unique_clusters = set(clusters)
                unique_clusters.discard(-1)  # Remove noise points
                
                for cluster_id in unique_clusters:
                    cluster_events = [geo_events[i] for i, c in enumerate(clusters) if c == cluster_id]
                    
                    if len(cluster_events) >= self.trend_thresholds['geo_cluster']:
                        # Calculate cluster center
                        cluster_coords = [event.location for event in cluster_events]
                        center_lat = np.mean([coord[0] for coord in cluster_coords])
                        center_lon = np.mean([coord[1] for coord in cluster_coords])
                        
                        # Determine event type based on keywords
                        all_keywords = []
                        for event in cluster_events:
                            if event.keywords:
                                all_keywords.extend(event.keywords)
                        
                        top_keywords = Counter(all_keywords).most_common(5)
                        event_type = top_keywords[0][0] if top_keywords else 'unknown'
                        
                        trend = TrendAlert(
                            alert_id=str(uuid.uuid4()),
                            trend_type='geo_cluster',
                            severity='medium' if len(cluster_events) < 10 else 'high',
                            description=f"Geographical cluster detected: {len(cluster_events)} events near ({center_lat:.2f}, {center_lon:.2f})",
                            affected_articles=[event.event_id for event in cluster_events],
                            confidence=min(1.0, len(cluster_events) / 20),
                            detected_at=datetime.now(),
                            location=(center_lat, center_lon),
                            metadata={
                                'cluster_size': len(cluster_events),
                                'event_type': event_type,
                                'top_keywords': top_keywords
                            }
                        )
                        trends.append(trend)
                        
                        # Record geographical event
                        GEO_EVENTS.labels(
                            country='unknown',  # Would need reverse geocoding
                            event_type=event_type
                        ).inc()
        
        except Exception as e:
            logger.error(f"Geo cluster detection failed: {e}")
        
        return trends
    
    async def _detect_emerging_topics(self, events: List[NewsEvent]) -> List[TrendAlert]:
        """Detect emerging topics using topic modeling."""
        trends = []
        
        try:
            # Prepare text data
            texts = []
            for event in events:
                if event.content:
                    texts.append(event.content)
            
            if len(texts) >= 10:  # Need sufficient data for topic modeling
                # Vectorize texts
                vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                doc_term_matrix = vectorizer.fit_transform(texts)
                
                # Perform topic modeling
                lda = LatentDirichletAllocation(
                    n_components=5,
                    random_state=42,
                    max_iter=10
                )
                lda.fit(doc_term_matrix)
                
                # Extract topics
                feature_names = vectorizer.get_feature_names_out()
                
                for topic_idx, topic in enumerate(lda.components_):
                    top_words_idx = topic.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    topic_strength = np.max(topic)
                    
                    # Check if this is an emerging topic
                    if topic_strength > self.trend_thresholds['topic_emergence']:
                        # Find documents belonging to this topic
                        doc_topic_probs = lda.transform(doc_term_matrix)
                        topic_docs = np.where(doc_topic_probs[:, topic_idx] > 0.3)[0]
                        
                        if len(topic_docs) >= 3:  # At least 3 documents
                            affected_articles = [events[i].event_id for i in topic_docs]
                            
                            trend = TrendAlert(
                                alert_id=str(uuid.uuid4()),
                                trend_type='emerging_topic',
                                severity='medium',
                                description=f"Emerging topic detected: {', '.join(top_words[:5])}",
                                affected_articles=affected_articles,
                                confidence=topic_strength,
                                detected_at=datetime.now(),
                                metadata={
                                    'topic_words': top_words,
                                    'topic_strength': topic_strength,
                                    'num_documents': len(topic_docs)
                                }
                            )
                            trends.append(trend)
        
        except Exception as e:
            logger.error(f"Emerging topic detection failed: {e}")
        
        return trends

class AnomalyDetector:
    """Advanced anomaly detection for news streams."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        
        # Anomaly detection models
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # Feature extractors
        self.scaler = StandardScaler()
        
        # Historical data for baseline
        self.feature_history = deque(maxlen=1000)
        self.model_trained = False
        
        logger.info("Anomaly detector initialized")
    
    async def detect_anomalies(self, events: List[NewsEvent]) -> List[TrendAlert]:
        """Detect anomalies in news events."""
        anomalies = []
        
        try:
            # Extract features from events
            features = self._extract_features(events)
            
            if len(features) > 0:
                # Update feature history
                self.feature_history.extend(features)
                
                # Train model if we have enough data
                if len(self.feature_history) >= 100 and not self.model_trained:
                    self._train_anomaly_model()
                
                # Detect anomalies if model is trained
                if self.model_trained:
                    anomaly_scores = self.isolation_forest.decision_function(features)
                    anomaly_labels = self.isolation_forest.predict(features)
                    
                    # Process anomalies
                    for i, (score, label) in enumerate(zip(anomaly_scores, anomaly_labels)):
                        if label == -1:  # Anomaly detected
                            event = events[i]
                            
                            # Determine anomaly severity
                            severity = 'low'
                            if score < -0.5:
                                severity = 'high'
                            elif score < -0.2:
                                severity = 'medium'
                            
                            anomaly = TrendAlert(
                                alert_id=str(uuid.uuid4()),
                                trend_type='anomaly',
                                severity=severity,
                                description=f"Anomalous news event detected: {event.title[:100]}...",
                                affected_articles=[event.event_id],
                                confidence=abs(score),
                                detected_at=datetime.now(),
                                metadata={
                                    'anomaly_score': score,
                                    'event_features': features[i].tolist()
                                }
                            )
                            anomalies.append(anomaly)
        
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
        
        return anomalies
    
    def _extract_features(self, events: List[NewsEvent]) -> np.ndarray:
        """Extract numerical features from news events."""
        features = []
        
        for event in events:
            feature_vector = [
                len(event.title) if event.title else 0,
                len(event.content) if event.content else 0,
                event.sentiment if event.sentiment is not None else 0,
                len(event.keywords) if event.keywords else 0,
                len(event.entities) if event.entities else 0,
                1 if event.location else 0,
                hash(event.source) % 1000,  # Source as numerical feature
                hash(event.category) % 100 if event.category else 0,
                event.timestamp.hour,  # Time of day
                event.timestamp.weekday(),  # Day of week
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _train_anomaly_model(self):
        """Train the anomaly detection model."""
        try:
            # Convert feature history to array
            feature_array = np.array(list(self.feature_history))
            
            # Scale features
            scaled_features = self.scaler.fit_transform(feature_array)
            
            # Train isolation forest
            self.isolation_forest.fit(scaled_features)
            
            self.model_trained = True
            logger.info("Anomaly detection model trained")
            
        except Exception as e:
            logger.error(f"Failed to train anomaly model: {e}")

def create_streaming_dashboard(stream_processor: StreamProcessor) -> dash.Dash:
    """Create real-time dashboard for stream monitoring."""
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Real-Time News Processing Dashboard", 
                style={'textAlign': 'center', 'marginBottom': 30}),
        
        # Key metrics row
        html.Div([
            html.Div([
                html.H3("Processing Rate"),
                html.H2(id="processing-rate", children="0 events/sec")
            ], className="metric-box", style={'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
            
            html.Div([
                html.H3("Total Processed"),
                html.H2(id="total-processed", children="0")
            ], className="metric-box", style={'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
            
            html.Div([
                html.H3("Active Trends"),
                html.H2(id="active-trends", children="0")
            ], className="metric-box", style={'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
            
            html.Div([
                html.H3("Anomalies"),
                html.H2(id="anomalies", children="0")
            ], className="metric-box", style={'width': '23%', 'display': 'inline-block', 'margin': '1%'})
        ]),
        
        # Charts row
        html.Div([
            html.Div([
                dcc.Graph(id="sentiment-timeline")
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id="category-distribution")
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        html.Div([
            html.Div([
                dcc.Graph(id="geographic-heatmap")
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id="keyword-trends")
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        # Auto-refresh
        dcc.Interval(
            id='interval-component',
            interval=5000,  # Update every 5 seconds
            n_intervals=0
        )
    ])
    
    @app.callback(
        [Output('processing-rate', 'children'),
         Output('total-processed', 'children'),
         Output('active-trends', 'children'),
         Output('anomalies', 'children')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_metrics(n):
        # Get current metrics from stream processor
        stats = stream_processor.processing_stats
        
        # Get trend and anomaly counts from Redis
        trend_count = stream_processor.redis_client.llen('trend_alerts')
        anomaly_count = stream_processor.redis_client.llen('anomaly_alerts')
        
        return (
            f"{stats['processing_rate']:.1f} events/sec",
            f"{stats['total_processed']:,}",
            str(trend_count),
            str(anomaly_count)
        )
    
    @app.callback(
        Output('sentiment-timeline', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_sentiment_timeline(n):
        # Query recent sentiment data
        # This would typically query your time-series database
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[datetime.now() - timedelta(minutes=i) for i in range(60, 0, -1)],
            y=np.random.normal(0, 0.3, 60),  # Mock data
            mode='lines',
            name='Average Sentiment'
        ))
        
        fig.update_layout(
            title='Sentiment Timeline (Last Hour)',
            xaxis_title='Time',
            yaxis_title='Sentiment Score',
            yaxis=dict(range=[-1, 1])
        )
        
        return fig
    
    @app.callback(
        Output('category-distribution', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_category_distribution(n):
        # Mock category data
        categories = ['Politics', 'Business', 'Technology', 'Health', 'Sports', 'Entertainment']
        values = np.random.randint(10, 100, len(categories))
        
        fig = go.Figure(data=[go.Pie(
            labels=categories,
            values=values,
            hole=0.3
        )])
        
        fig.update_layout(title='News Category Distribution')
        
        return fig
    
    return app

def main():
    """Main function to run the streaming pipeline."""
    # Configuration
    config = StreamingConfig(
        kafka_bootstrap_servers=['localhost:9092'],
        kafka_topics={
            'raw_news': 'raw_news',
            'processed_news': 'processed_news',
            'trend_alerts': 'trend_alerts',
            'anomaly_alerts': 'anomaly_alerts'
        },
        redis_host='localhost',
        redis_port=6379,
        elasticsearch_host='localhost',
        elasticsearch_port=9200,
        mongodb_uri='mongodb://localhost:27017/',
        postgres_uri='postgresql://localhost:5432/news_pipeline',
        batch_size=50,
        processing_timeout=10,
        max_workers=4,
        enable_gpu=torch.cuda.is_available(),
        languages=['en', 'es', 'fr', 'de']
    )
    
    # Start Prometheus metrics server
    start_http_server(8002)
    
    # Initialize and start stream processor
    processor = StreamProcessor(config)
    processor.start()
    
    # Create and run dashboard
    dashboard = create_streaming_dashboard(processor)
    
    logger.info("Real-time news processing pipeline started")
    logger.info("Dashboard available at http://localhost:8050")
    logger.info("Metrics available at http://localhost:8002")
    
    try:
        dashboard.run_server(host='0.0.0.0', port=8050, debug=False)
    except KeyboardInterrupt:
        logger.info("Shutting down pipeline")
        processor.stop()

if __name__ == "__main__":
    main()