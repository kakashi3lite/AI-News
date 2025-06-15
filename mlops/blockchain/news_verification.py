#!/usr/bin/env python3
"""
Dr. NewsForge's Blockchain-Based News Verification & Trust System

Implements a decentralized news verification system using blockchain technology
to combat misinformation and establish trust scores for news sources.

Features:
- Blockchain-based news verification
- Decentralized fact-checking network
- Source credibility scoring
- Immutable audit trails
- Smart contracts for verification rewards
- Cross-platform verification consensus
- Real-time misinformation detection
- Transparent trust metrics

Author: Dr. Nova "NewsForge" Arclight
Version: 2.0.0
"""

import os
import json
import time
import logging
import asyncio
import hashlib
import threading
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import uuid
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertModel
)
from sentence_transformers import SentenceTransformer
import spacy
from textblob import TextBlob

import web3
from web3 import Web3
from eth_account import Account
from solcx import compile_source, install_solc
import ipfshttpclient
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key

import requests
from flask import Flask, request, jsonify, Response
import redis
import pymongo
from pymongo import MongoClient
from elasticsearch import Elasticsearch

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import mlflow
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
VERIFICATION_REQUESTS = Counter('verification_requests_total', 'Total verification requests', ['source_type'])
VERIFICATION_LATENCY = Histogram('verification_latency_seconds', 'Verification processing latency')
TRUST_SCORE_UPDATES = Counter('trust_score_updates_total', 'Trust score updates', ['source'])
MISINFORMATION_DETECTED = Counter('misinformation_detected_total', 'Misinformation cases detected', ['severity'])
BLOCKCHAIN_TRANSACTIONS = Counter('blockchain_transactions_total', 'Blockchain transactions', ['type'])
CONSENSUS_ROUNDS = Counter('consensus_rounds_total', 'Consensus rounds completed', ['result'])
VERIFIER_REWARDS = Counter('verifier_rewards_total', 'Rewards distributed to verifiers')
NETWORK_TRUST = Gauge('network_trust_score', 'Overall network trust score')

@dataclass
class NewsArticle:
    """Represents a news article for verification."""
    article_id: str
    title: str
    content: str
    source: str
    author: str
    published_at: datetime
    url: str
    hash: str
    language: str = 'en'
    category: Optional[str] = None
    claims: Optional[List[str]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    embedding: Optional[List[float]] = None

@dataclass
class VerificationResult:
    """Represents the result of news verification."""
    verification_id: str
    article_id: str
    verifier_id: str
    trust_score: float  # 0.0 to 1.0
    credibility_score: float  # 0.0 to 1.0
    misinformation_probability: float  # 0.0 to 1.0
    fact_check_results: List[Dict[str, Any]]
    source_analysis: Dict[str, Any]
    consensus_score: float
    verification_timestamp: datetime
    blockchain_hash: Optional[str] = None
    ipfs_hash: Optional[str] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    confidence: float = 0.0

@dataclass
class SourceCredibility:
    """Represents source credibility metrics."""
    source_id: str
    source_name: str
    domain: str
    trust_score: float
    accuracy_history: List[float]
    bias_score: float  # -1.0 (left) to 1.0 (right)
    factual_reporting: str  # 'high', 'mixed', 'low'
    verification_count: int
    last_updated: datetime
    reputation_factors: Dict[str, float]

@dataclass
class VerifierNode:
    """Represents a verifier node in the network."""
    node_id: str
    public_key: str
    reputation_score: float
    verification_count: int
    accuracy_rate: float
    stake_amount: float
    last_active: datetime
    specializations: List[str]  # Topics/categories
    geographic_region: str
    node_type: str  # 'human', 'ai', 'hybrid'

class BlockchainVerificationSystem:
    """Core blockchain verification system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Blockchain setup
        self.w3 = Web3(Web3.HTTPProvider(config.get('ethereum_rpc', 'http://localhost:8545')))
        self.account = Account.from_key(config.get('private_key'))
        
        # IPFS setup
        try:
            self.ipfs_client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')
        except Exception as e:
            logger.warning(f"IPFS connection failed: {e}")
            self.ipfs_client = None
        
        # Smart contract
        self.contract = None
        self.deploy_smart_contract()
        
        # Storage backends
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            decode_responses=True
        )
        
        self.mongo_client = MongoClient(config.get('mongodb_uri', 'mongodb://localhost:27017/'))
        self.db = self.mongo_client.news_verification
        
        # Verification components
        self.fact_checker = FactChecker(config)
        self.source_analyzer = SourceAnalyzer(config)
        self.consensus_engine = ConsensusEngine(config)
        
        # Network state
        self.verifier_nodes = {}
        self.source_credibility = {}
        self.verification_history = deque(maxlen=10000)
        
        logger.info("Blockchain verification system initialized")
    
    def deploy_smart_contract(self):
        """Deploy the news verification smart contract."""
        try:
            # Smart contract source code
            contract_source = '''
            pragma solidity ^0.8.0;
            
            contract NewsVerification {
                struct VerificationRecord {
                    string articleHash;
                    address verifier;
                    uint256 trustScore;
                    uint256 timestamp;
                    string ipfsHash;
                    bool isValid;
                }
                
                struct SourceCredibility {
                    string sourceName;
                    uint256 trustScore;
                    uint256 verificationCount;
                    uint256 lastUpdated;
                }
                
                mapping(string => VerificationRecord[]) public verifications;
                mapping(string => SourceCredibility) public sources;
                mapping(address => uint256) public verifierReputations;
                mapping(address => uint256) public verifierStakes;
                
                event VerificationSubmitted(string articleHash, address verifier, uint256 trustScore);
                event SourceUpdated(string sourceName, uint256 newTrustScore);
                event RewardDistributed(address verifier, uint256 amount);
                
                function submitVerification(
                    string memory articleHash,
                    uint256 trustScore,
                    string memory ipfsHash
                ) public {
                    require(verifierStakes[msg.sender] > 0, "Verifier must have stake");
                    require(trustScore <= 100, "Trust score must be <= 100");
                    
                    VerificationRecord memory record = VerificationRecord({
                        articleHash: articleHash,
                        verifier: msg.sender,
                        trustScore: trustScore,
                        timestamp: block.timestamp,
                        ipfsHash: ipfsHash,
                        isValid: true
                    });
                    
                    verifications[articleHash].push(record);
                    emit VerificationSubmitted(articleHash, msg.sender, trustScore);
                }
                
                function updateSourceCredibility(
                    string memory sourceName,
                    uint256 newTrustScore
                ) public {
                    sources[sourceName].trustScore = newTrustScore;
                    sources[sourceName].verificationCount += 1;
                    sources[sourceName].lastUpdated = block.timestamp;
                    
                    emit SourceUpdated(sourceName, newTrustScore);
                }
                
                function stakeTokens() public payable {
                    require(msg.value > 0, "Stake must be positive");
                    verifierStakes[msg.sender] += msg.value;
                }
                
                function distributeReward(address verifier, uint256 amount) public {
                    require(verifierStakes[verifier] > 0, "Invalid verifier");
                    
                    verifierReputations[verifier] += amount;
                    emit RewardDistributed(verifier, amount);
                }
                
                function getVerifications(string memory articleHash) 
                    public view returns (VerificationRecord[] memory) {
                    return verifications[articleHash];
                }
                
                function getSourceCredibility(string memory sourceName) 
                    public view returns (SourceCredibility memory) {
                    return sources[sourceName];
                }
            }
            '''
            
            # Compile contract
            install_solc('0.8.0')
            compiled_sol = compile_source(contract_source)
            contract_interface = compiled_sol['<stdin>:NewsVerification']
            
            # Deploy contract
            if self.w3.isConnected():
                contract = self.w3.eth.contract(
                    abi=contract_interface['abi'],
                    bytecode=contract_interface['bin']
                )
                
                # Build transaction
                transaction = contract.constructor().buildTransaction({
                    'from': self.account.address,
                    'nonce': self.w3.eth.get_transaction_count(self.account.address),
                    'gas': 2000000,
                    'gasPrice': self.w3.toWei('20', 'gwei')
                })
                
                # Sign and send transaction
                signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.privateKey)
                tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
                
                # Wait for transaction receipt
                tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                
                # Create contract instance
                self.contract = self.w3.eth.contract(
                    address=tx_receipt.contractAddress,
                    abi=contract_interface['abi']
                )
                
                logger.info(f"Smart contract deployed at: {tx_receipt.contractAddress}")
            else:
                logger.warning("Ethereum node not connected, using mock contract")
                
        except Exception as e:
            logger.error(f"Smart contract deployment failed: {e}")
    
    async def verify_article(self, article: NewsArticle) -> VerificationResult:
        """Verify a news article using the blockchain network."""
        start_time = time.time()
        
        try:
            # Generate verification ID
            verification_id = str(uuid.uuid4())
            
            # Perform fact-checking
            fact_check_results = await self.fact_checker.check_facts(article)
            
            # Analyze source credibility
            source_analysis = await self.source_analyzer.analyze_source(article.source)
            
            # Calculate initial scores
            trust_score = self._calculate_trust_score(fact_check_results, source_analysis)
            credibility_score = source_analysis.get('credibility_score', 0.5)
            misinformation_prob = self._calculate_misinformation_probability(fact_check_results)
            
            # Store verification data in IPFS
            ipfs_hash = None
            if self.ipfs_client:
                verification_data = {
                    'article': asdict(article),
                    'fact_check_results': fact_check_results,
                    'source_analysis': source_analysis,
                    'timestamp': datetime.now().isoformat()
                }
                ipfs_result = self.ipfs_client.add_json(verification_data)
                ipfs_hash = ipfs_result['Hash']
            
            # Submit to blockchain
            blockchain_hash = await self._submit_to_blockchain(
                article.hash, trust_score, ipfs_hash
            )
            
            # Get consensus from network
            consensus_score = await self.consensus_engine.get_consensus(
                article, fact_check_results, source_analysis
            )
            
            # Create verification result
            result = VerificationResult(
                verification_id=verification_id,
                article_id=article.article_id,
                verifier_id=self.account.address,
                trust_score=trust_score,
                credibility_score=credibility_score,
                misinformation_probability=misinformation_prob,
                fact_check_results=fact_check_results,
                source_analysis=source_analysis,
                consensus_score=consensus_score,
                verification_timestamp=datetime.now(),
                blockchain_hash=blockchain_hash,
                ipfs_hash=ipfs_hash,
                confidence=min(consensus_score, trust_score)
            )
            
            # Store in database
            await self._store_verification_result(result)
            
            # Update source credibility
            await self._update_source_credibility(article.source, result)
            
            # Record metrics
            processing_time = time.time() - start_time
            VERIFICATION_LATENCY.observe(processing_time)
            VERIFICATION_REQUESTS.labels(source_type=article.source).inc()
            
            if misinformation_prob > 0.7:
                MISINFORMATION_DETECTED.labels(severity='high').inc()
            elif misinformation_prob > 0.4:
                MISINFORMATION_DETECTED.labels(severity='medium').inc()
            
            logger.info(f"Article verification completed: {verification_id}")
            return result
            
        except Exception as e:
            logger.error(f"Article verification failed: {e}")
            raise
    
    def _calculate_trust_score(self, fact_check_results: List[Dict], source_analysis: Dict) -> float:
        """Calculate trust score based on fact-checking and source analysis."""
        try:
            # Fact-checking component (60% weight)
            fact_score = 0.0
            if fact_check_results:
                verified_claims = sum(1 for result in fact_check_results if result.get('verified', False))
                total_claims = len(fact_check_results)
                fact_score = verified_claims / total_claims if total_claims > 0 else 0.5
            
            # Source credibility component (40% weight)
            source_score = source_analysis.get('credibility_score', 0.5)
            
            # Weighted combination
            trust_score = (fact_score * 0.6) + (source_score * 0.4)
            
            return max(0.0, min(1.0, trust_score))
            
        except Exception as e:
            logger.error(f"Trust score calculation failed: {e}")
            return 0.5
    
    def _calculate_misinformation_probability(self, fact_check_results: List[Dict]) -> float:
        """Calculate probability of misinformation."""
        try:
            if not fact_check_results:
                return 0.5  # Neutral when no fact-check data
            
            false_claims = sum(1 for result in fact_check_results if not result.get('verified', True))
            total_claims = len(fact_check_results)
            
            # Higher false claim ratio = higher misinformation probability
            misinformation_prob = false_claims / total_claims if total_claims > 0 else 0.0
            
            return max(0.0, min(1.0, misinformation_prob))
            
        except Exception as e:
            logger.error(f"Misinformation probability calculation failed: {e}")
            return 0.5
    
    async def _submit_to_blockchain(self, article_hash: str, trust_score: float, ipfs_hash: str) -> Optional[str]:
        """Submit verification to blockchain."""
        try:
            if not self.contract:
                return None
            
            # Convert trust score to integer (0-100)
            trust_score_int = int(trust_score * 100)
            
            # Build transaction
            transaction = self.contract.functions.submitVerification(
                article_hash,
                trust_score_int,
                ipfs_hash or ""
            ).buildTransaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 200000,
                'gasPrice': self.w3.toWei('20', 'gwei')
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.privateKey)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            BLOCKCHAIN_TRANSACTIONS.labels(type='verification').inc()
            
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Blockchain submission failed: {e}")
            return None
    
    async def _store_verification_result(self, result: VerificationResult):
        """Store verification result in database."""
        try:
            # Store in MongoDB
            doc = asdict(result)
            doc['_id'] = result.verification_id
            self.db.verifications.insert_one(doc)
            
            # Cache in Redis
            self.redis_client.setex(
                f"verification:{result.verification_id}",
                3600,  # 1 hour TTL
                json.dumps(doc, default=str)
            )
            
            # Add to history
            self.verification_history.append(result)
            
        except Exception as e:
            logger.error(f"Failed to store verification result: {e}")
    
    async def _update_source_credibility(self, source: str, result: VerificationResult):
        """Update source credibility based on verification result."""
        try:
            # Get existing credibility data
            existing = self.source_credibility.get(source)
            
            if existing:
                # Update existing credibility
                existing.accuracy_history.append(result.trust_score)
                existing.trust_score = np.mean(existing.accuracy_history[-50:])  # Last 50 verifications
                existing.verification_count += 1
                existing.last_updated = datetime.now()
            else:
                # Create new credibility record
                domain = source.split('/')[-1] if '/' in source else source
                existing = SourceCredibility(
                    source_id=str(uuid.uuid4()),
                    source_name=source,
                    domain=domain,
                    trust_score=result.trust_score,
                    accuracy_history=[result.trust_score],
                    bias_score=0.0,  # Would be calculated separately
                    factual_reporting='mixed',
                    verification_count=1,
                    last_updated=datetime.now(),
                    reputation_factors={}
                )
            
            self.source_credibility[source] = existing
            
            # Update in blockchain
            if self.contract:
                await self._update_blockchain_source_credibility(source, existing.trust_score)
            
            # Store in database
            doc = asdict(existing)
            self.db.source_credibility.replace_one(
                {'source_name': source},
                doc,
                upsert=True
            )
            
            TRUST_SCORE_UPDATES.labels(source=source).inc()
            
        except Exception as e:
            logger.error(f"Failed to update source credibility: {e}")
    
    async def _update_blockchain_source_credibility(self, source: str, trust_score: float):
        """Update source credibility on blockchain."""
        try:
            if not self.contract:
                return
            
            trust_score_int = int(trust_score * 100)
            
            transaction = self.contract.functions.updateSourceCredibility(
                source,
                trust_score_int
            ).buildTransaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 100000,
                'gasPrice': self.w3.toWei('20', 'gwei')
            })
            
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.privateKey)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            BLOCKCHAIN_TRANSACTIONS.labels(type='source_update').inc()
            
        except Exception as e:
            logger.error(f"Blockchain source update failed: {e}")
    
    def get_source_credibility(self, source: str) -> Optional[SourceCredibility]:
        """Get source credibility information."""
        return self.source_credibility.get(source)
    
    def get_verification_history(self, article_id: str) -> List[VerificationResult]:
        """Get verification history for an article."""
        return [
            result for result in self.verification_history
            if result.article_id == article_id
        ]

class FactChecker:
    """AI-powered fact-checking system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load fact-checking models
        self.claim_detector = pipeline(
            "text-classification",
            model="microsoft/DialoGPT-medium",  # Placeholder - would use specialized model
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.fact_verification_model = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Knowledge bases
        self.knowledge_sources = [
            'https://api.factcheck.org',
            'https://www.snopes.com/api',
            'https://www.politifact.com/api'
        ]
        
        # NLP components
        self.nlp = spacy.load('en_core_web_sm')
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("Fact checker initialized")
    
    async def check_facts(self, article: NewsArticle) -> List[Dict[str, Any]]:
        """Perform comprehensive fact-checking on article."""
        try:
            # Extract claims from article
            claims = await self._extract_claims(article.content)
            
            # Verify each claim
            fact_check_results = []
            for claim in claims:
                result = await self._verify_claim(claim, article)
                fact_check_results.append(result)
            
            return fact_check_results
            
        except Exception as e:
            logger.error(f"Fact checking failed: {e}")
            return []
    
    async def _extract_claims(self, content: str) -> List[str]:
        """Extract factual claims from article content."""
        try:
            # Use NLP to identify claim-like sentences
            doc = self.nlp(content)
            
            claims = []
            for sent in doc.sents:
                # Look for sentences with factual indicators
                if self._is_factual_claim(sent.text):
                    claims.append(sent.text.strip())
            
            return claims[:10]  # Limit to top 10 claims
            
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            return []
    
    def _is_factual_claim(self, sentence: str) -> bool:
        """Determine if a sentence contains a factual claim."""
        # Simple heuristics for factual claims
        factual_indicators = [
            'according to', 'study shows', 'research indicates',
            'data reveals', 'statistics show', 'report states',
            'announced', 'confirmed', 'revealed', 'discovered'
        ]
        
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in factual_indicators)
    
    async def _verify_claim(self, claim: str, article: NewsArticle) -> Dict[str, Any]:
        """Verify a single factual claim."""
        try:
            # Search knowledge bases
            knowledge_results = await self._search_knowledge_bases(claim)
            
            # Use AI model for verification
            ai_verification = await self._ai_verify_claim(claim, knowledge_results)
            
            # Cross-reference with reliable sources
            source_verification = await self._cross_reference_sources(claim)
            
            # Combine results
            verified = (
                ai_verification.get('verified', False) and
                source_verification.get('verified', False)
            )
            
            confidence = min(
                ai_verification.get('confidence', 0.0),
                source_verification.get('confidence', 0.0)
            )
            
            return {
                'claim': claim,
                'verified': verified,
                'confidence': confidence,
                'sources': knowledge_results,
                'ai_analysis': ai_verification,
                'source_analysis': source_verification,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Claim verification failed: {e}")
            return {
                'claim': claim,
                'verified': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def _search_knowledge_bases(self, claim: str) -> List[Dict[str, Any]]:
        """Search external knowledge bases for claim verification."""
        results = []
        
        try:
            # Search each knowledge source
            for source_url in self.knowledge_sources:
                try:
                    # Mock API call - would implement actual API integration
                    response = requests.get(
                        f"{source_url}/search",
                        params={'q': claim[:200]},  # Limit query length
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        results.append({
                            'source': source_url,
                            'results': data.get('results', []),
                            'confidence': data.get('confidence', 0.0)
                        })
                        
                except Exception as e:
                    logger.warning(f"Knowledge base search failed for {source_url}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return []
    
    async def _ai_verify_claim(self, claim: str, knowledge_results: List[Dict]) -> Dict[str, Any]:
        """Use AI model to verify claim against knowledge."""
        try:
            # Prepare context from knowledge results
            context = ""
            for result in knowledge_results:
                for item in result.get('results', [])[:3:  # Top 3 results per source
                    context += f"{item.get('text', '')} "
            
            if not context.strip():
                return {'verified': False, 'confidence': 0.0, 'reason': 'No context available'}
            
            # Use BART for natural language inference
            premise = context[:1000]  # Limit context length
            hypothesis = claim
            
            result = self.fact_verification_model(
                f"{premise} {self.fact_verification_model.tokenizer.sep_token} {hypothesis}"
            )
            
            # Interpret result
            label = result['label'].lower()
            confidence = result['score']
            
            verified = label == 'entailment'
            
            return {
                'verified': verified,
                'confidence': confidence,
                'label': label,
                'model_output': result
            }
            
        except Exception as e:
            logger.error(f"AI claim verification failed: {e}")
            return {'verified': False, 'confidence': 0.0, 'error': str(e)}
    
    async def _cross_reference_sources(self, claim: str) -> Dict[str, Any]:
        """Cross-reference claim with multiple reliable sources."""
        try:
            # Mock implementation - would integrate with news APIs
            reliable_sources = [
                'reuters.com', 'ap.org', 'bbc.com',
                'npr.org', 'pbs.org'
            ]
            
            supporting_sources = 0
            total_sources = len(reliable_sources)
            
            # Simulate source checking
            for source in reliable_sources:
                # Mock API call
                if hash(claim + source) % 3 == 0:  # Random simulation
                    supporting_sources += 1
            
            confidence = supporting_sources / total_sources
            verified = confidence > 0.5
            
            return {
                'verified': verified,
                'confidence': confidence,
                'supporting_sources': supporting_sources,
                'total_sources': total_sources
            }
            
        except Exception as e:
            logger.error(f"Source cross-reference failed: {e}")
            return {'verified': False, 'confidence': 0.0, 'error': str(e)}

class SourceAnalyzer:
    """Analyzes news source credibility and bias."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load source credibility database
        self.credibility_db = self._load_credibility_database()
        
        # Bias detection model
        self.bias_detector = pipeline(
            "text-classification",
            model="unitary/toxic-bert",  # Placeholder - would use bias-specific model
            device=0 if torch.cuda.is_available() else -1
        )
        
        logger.info("Source analyzer initialized")
    
    def _load_credibility_database(self) -> Dict[str, Dict]:
        """Load known source credibility ratings."""
        # Mock database - would load from actual credibility databases
        return {
            'reuters.com': {'trust_score': 0.95, 'bias': 0.0, 'factual': 'high'},
            'ap.org': {'trust_score': 0.94, 'bias': 0.0, 'factual': 'high'},
            'bbc.com': {'trust_score': 0.90, 'bias': -0.1, 'factual': 'high'},
            'cnn.com': {'trust_score': 0.75, 'bias': -0.3, 'factual': 'mixed'},
            'foxnews.com': {'trust_score': 0.70, 'bias': 0.4, 'factual': 'mixed'},
            'breitbart.com': {'trust_score': 0.30, 'bias': 0.8, 'factual': 'low'},
            'infowars.com': {'trust_score': 0.10, 'bias': 0.9, 'factual': 'low'}
        }
    
    async def analyze_source(self, source: str) -> Dict[str, Any]:
        """Analyze source credibility and characteristics."""
        try:
            # Extract domain from URL
            domain = self._extract_domain(source)
            
            # Get known credibility data
            known_data = self.credibility_db.get(domain, {})
            
            # Analyze domain characteristics
            domain_analysis = await self._analyze_domain(domain)
            
            # Combine analyses
            credibility_score = known_data.get('trust_score', domain_analysis.get('credibility_score', 0.5))
            bias_score = known_data.get('bias', domain_analysis.get('bias_score', 0.0))
            factual_reporting = known_data.get('factual', domain_analysis.get('factual_reporting', 'unknown'))
            
            return {
                'domain': domain,
                'credibility_score': credibility_score,
                'bias_score': bias_score,
                'factual_reporting': factual_reporting,
                'known_source': domain in self.credibility_db,
                'domain_analysis': domain_analysis,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Source analysis failed: {e}")
            return {
                'credibility_score': 0.5,
                'bias_score': 0.0,
                'factual_reporting': 'unknown',
                'error': str(e)
            }
    
    def _extract_domain(self, source: str) -> str:
        """Extract domain from source URL or name."""
        try:
            if source.startswith('http'):
                from urllib.parse import urlparse
                return urlparse(source).netloc.lower()
            else:
                return source.lower()
        except:
            return source.lower()
    
    async def _analyze_domain(self, domain: str) -> Dict[str, Any]:
        """Analyze domain characteristics for credibility assessment."""
        try:
            analysis = {
                'credibility_score': 0.5,
                'bias_score': 0.0,
                'factual_reporting': 'unknown'
            }
            
            # Domain age and reputation heuristics
            if any(indicator in domain for indicator in ['.gov', '.edu', '.org']):
                analysis['credibility_score'] += 0.2
            
            if any(indicator in domain for indicator in ['news', 'times', 'post', 'herald']):
                analysis['credibility_score'] += 0.1
            
            if any(indicator in domain for indicator in ['blog', 'wordpress', 'tumblr']):
                analysis['credibility_score'] -= 0.2
            
            # Suspicious domain patterns
            if any(indicator in domain for indicator in ['fake', 'conspiracy', 'truth']):
                analysis['credibility_score'] -= 0.3
                analysis['factual_reporting'] = 'low'
            
            # Ensure score bounds
            analysis['credibility_score'] = max(0.0, min(1.0, analysis['credibility_score']))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Domain analysis failed: {e}")
            return {'credibility_score': 0.5, 'bias_score': 0.0, 'factual_reporting': 'unknown'}

class ConsensusEngine:
    """Manages consensus among verifier nodes."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Consensus parameters
        self.min_verifiers = config.get('min_verifiers', 3)
        self.consensus_threshold = config.get('consensus_threshold', 0.67)
        self.timeout_seconds = config.get('consensus_timeout', 30)
        
        # Active verifier nodes
        self.verifier_nodes = {}
        
        logger.info("Consensus engine initialized")
    
    async def get_consensus(self, article: NewsArticle, fact_results: List[Dict], source_analysis: Dict) -> float:
        """Get consensus score from verifier network."""
        try:
            # Simulate consensus from multiple verifiers
            verifier_scores = []
            
            # Add our own verification
            our_score = self._calculate_verification_score(fact_results, source_analysis)
            verifier_scores.append(our_score)
            
            # Simulate other verifiers (in real implementation, would query network)
            for i in range(self.min_verifiers - 1):
                # Add some noise to simulate different verifier opinions
                noise = np.random.normal(0, 0.1)
                simulated_score = max(0.0, min(1.0, our_score + noise))
                verifier_scores.append(simulated_score)
            
            # Calculate consensus
            if len(verifier_scores) >= self.min_verifiers:
                consensus_score = np.mean(verifier_scores)
                agreement_level = 1.0 - np.std(verifier_scores)
                
                # Adjust consensus based on agreement level
                final_consensus = consensus_score * agreement_level
                
                CONSENSUS_ROUNDS.labels(result='success').inc()
                
                return max(0.0, min(1.0, final_consensus))
            else:
                CONSENSUS_ROUNDS.labels(result='insufficient_verifiers').inc()
                return our_score
            
        except Exception as e:
            logger.error(f"Consensus calculation failed: {e}")
            CONSENSUS_ROUNDS.labels(result='error').inc()
            return 0.5
    
    def _calculate_verification_score(self, fact_results: List[Dict], source_analysis: Dict) -> float:
        """Calculate verification score from fact-checking and source analysis."""
        try:
            # Fact-checking component
            fact_score = 0.5
            if fact_results:
                verified_count = sum(1 for result in fact_results if result.get('verified', False))
                fact_score = verified_count / len(fact_results)
            
            # Source credibility component
            source_score = source_analysis.get('credibility_score', 0.5)
            
            # Weighted combination
            verification_score = (fact_score * 0.7) + (source_score * 0.3)
            
            return max(0.0, min(1.0, verification_score))
            
        except Exception as e:
            logger.error(f"Verification score calculation failed: {e}")
            return 0.5

def create_verification_api(blockchain_system: BlockchainVerificationSystem) -> Flask:
    """Create Flask API for news verification."""
    app = Flask(__name__)
    
    @app.route('/verify', methods=['POST'])
    async def verify_news():
        try:
            data = request.get_json()
            
            # Create NewsArticle object
            article = NewsArticle(
                article_id=data.get('article_id', str(uuid.uuid4())),
                title=data.get('title', ''),
                content=data.get('content', ''),
                source=data.get('source', ''),
                author=data.get('author', ''),
                published_at=datetime.fromisoformat(data.get('published_at', datetime.now().isoformat())),
                url=data.get('url', ''),
                hash=hashlib.sha256(data.get('content', '').encode()).hexdigest(),
                language=data.get('language', 'en')
            )
            
            # Verify article
            result = await blockchain_system.verify_article(article)
            
            return jsonify({
                'verification_id': result.verification_id,
                'trust_score': result.trust_score,
                'credibility_score': result.credibility_score,
                'misinformation_probability': result.misinformation_probability,
                'consensus_score': result.consensus_score,
                'blockchain_hash': result.blockchain_hash,
                'ipfs_hash': result.ipfs_hash,
                'verification_timestamp': result.verification_timestamp.isoformat()
            })
            
        except Exception as e:
            logger.error(f"Verification API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/source/<path:source_name>', methods=['GET'])
    def get_source_credibility(source_name):
        try:
            credibility = blockchain_system.get_source_credibility(source_name)
            
            if credibility:
                return jsonify(asdict(credibility))
            else:
                return jsonify({'error': 'Source not found'}), 404
                
        except Exception as e:
            logger.error(f"Source credibility API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/verification/<verification_id>', methods=['GET'])
    def get_verification(verification_id):
        try:
            # Get from Redis cache first
            cached = blockchain_system.redis_client.get(f"verification:{verification_id}")
            
            if cached:
                return jsonify(json.loads(cached))
            
            # Get from MongoDB
            result = blockchain_system.db.verifications.find_one({'_id': verification_id})
            
            if result:
                result.pop('_id', None)  # Remove MongoDB ID
                return jsonify(result)
            else:
                return jsonify({'error': 'Verification not found'}), 404
                
        except Exception as e:
            logger.error(f"Get verification API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'blockchain_connected': blockchain_system.w3.isConnected() if blockchain_system.w3 else False,
            'contract_deployed': blockchain_system.contract is not None
        })
    
    return app

def main():
    """Main function to run the blockchain verification system."""
    # Configuration
    config = {
        'ethereum_rpc': os.getenv('ETHEREUM_RPC', 'http://localhost:8545'),
        'private_key': os.getenv('PRIVATE_KEY', '0x' + '0' * 64),  # Use proper private key
        'redis_host': os.getenv('REDIS_HOST', 'localhost'),
        'redis_port': int(os.getenv('REDIS_PORT', 6379)),
        'mongodb_uri': os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'),
        'min_verifiers': 3,
        'consensus_threshold': 0.67,
        'consensus_timeout': 30
    }
    
    # Start Prometheus metrics server
    start_http_server(8003)
    
    # Initialize blockchain verification system
    blockchain_system = BlockchainVerificationSystem(config)
    
    # Create and run API
    app = create_verification_api(blockchain_system)
    
    logger.info("Blockchain news verification system started")
    logger.info("API available at http://localhost:5001")
    logger.info("Metrics available at http://localhost:8003")
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False)
    except KeyboardInterrupt:
        logger.info("Shutting down verification system")

if __name__ == "__main__":
    main()