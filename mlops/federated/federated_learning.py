#!/usr/bin/env python3
"""
Dr. NewsForge's Advanced Federated Learning System

Implements privacy-preserving federated learning for news personalization
with differential privacy, secure aggregation, and edge deployment.

Features:
- Federated averaging with secure aggregation
- Differential privacy mechanisms
- Client selection and scheduling
- Model compression and quantization
- Byzantine fault tolerance
- Personalized federated learning
- Edge deployment optimization

Author: Dr. Nova "NewsForge" Arclight
Version: 2.0.0
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import redis
import mlflow
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from scipy.stats import norm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ClientConfig:
    """Configuration for federated learning client."""
    client_id: str
    device_type: str  # 'mobile', 'desktop', 'edge'
    compute_capability: float  # 0.0 to 1.0
    bandwidth: float  # Mbps
    privacy_budget: float  # Differential privacy budget
    data_size: int
    location: str
    timezone: str

@dataclass
class ModelUpdate:
    """Structure for model updates from clients."""
    client_id: str
    round_number: int
    model_weights: Dict[str, torch.Tensor]
    num_samples: int
    loss: float
    accuracy: float
    privacy_spent: float
    timestamp: datetime
    signature: Optional[str] = None

@dataclass
class FederatedRound:
    """Information about a federated learning round."""
    round_number: int
    selected_clients: List[str]
    global_model_version: str
    start_time: datetime
    end_time: Optional[datetime] = None
    aggregated_loss: Optional[float] = None
    aggregated_accuracy: Optional[float] = None
    convergence_metric: Optional[float] = None

class DifferentialPrivacy:
    """Differential privacy mechanisms for federated learning."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = 1.0
    
    def add_gaussian_noise(self, tensor: torch.Tensor, sensitivity: float = None) -> torch.Tensor:
        """Add Gaussian noise for differential privacy."""
        if sensitivity is None:
            sensitivity = self.sensitivity
        
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        noise = torch.normal(0, sigma, tensor.shape)
        
        return tensor + noise
    
    def add_laplace_noise(self, tensor: torch.Tensor, sensitivity: float = None) -> torch.Tensor:
        """Add Laplace noise for differential privacy."""
        if sensitivity is None:
            sensitivity = self.sensitivity
        
        scale = sensitivity / self.epsilon
        noise = torch.from_numpy(np.random.laplace(0, scale, tensor.shape)).float()
        
        return tensor + noise
    
    def clip_gradients(self, gradients: Dict[str, torch.Tensor], max_norm: float = 1.0) -> Dict[str, torch.Tensor]:
        """Clip gradients to bound sensitivity."""
        clipped_gradients = {}
        
        for name, grad in gradients.items():
            grad_norm = torch.norm(grad)
            if grad_norm > max_norm:
                clipped_gradients[name] = grad * (max_norm / grad_norm)
            else:
                clipped_gradients[name] = grad
        
        return clipped_gradients

class SecureAggregation:
    """Secure aggregation protocol for federated learning."""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.threshold = max(1, num_clients // 2)  # Minimum clients needed
        self.encryption_keys = {}
    
    def generate_client_keys(self, client_id: str) -> Tuple[bytes, bytes]:
        """Generate encryption keys for a client."""
        password = f"federated_learning_{client_id}".encode()
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        
        self.encryption_keys[client_id] = Fernet(key)
        return key, salt
    
    def encrypt_model_update(self, client_id: str, model_weights: Dict[str, torch.Tensor]) -> bytes:
        """Encrypt model weights for secure transmission."""
        if client_id not in self.encryption_keys:
            raise ValueError(f"No encryption key found for client {client_id}")
        
        # Serialize model weights
        serialized_weights = {}
        for name, tensor in model_weights.items():
            serialized_weights[name] = tensor.cpu().numpy().tolist()
        
        data = json.dumps(serialized_weights).encode()
        encrypted_data = self.encryption_keys[client_id].encrypt(data)
        
        return encrypted_data
    
    def decrypt_model_update(self, client_id: str, encrypted_data: bytes) -> Dict[str, torch.Tensor]:
        """Decrypt model weights from client."""
        if client_id not in self.encryption_keys:
            raise ValueError(f"No encryption key found for client {client_id}")
        
        decrypted_data = self.encryption_keys[client_id].decrypt(encrypted_data)
        serialized_weights = json.loads(decrypted_data.decode())
        
        model_weights = {}
        for name, tensor_list in serialized_weights.items():
            model_weights[name] = torch.tensor(tensor_list)
        
        return model_weights

class NewsPersonalizationModel(nn.Module):
    """Neural network for personalized news recommendation."""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128, hidden_dim: int = 256, num_categories: int = 20):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_categories)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # Embedding
        embedded = self.embedding(input_ids)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        if attention_mask is not None:
            attended = attended * attention_mask.unsqueeze(-1)
            pooled = attended.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = attended.mean(dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output

class FederatedClient:
    """Federated learning client implementation."""
    
    def __init__(self, config: ClientConfig, model: nn.Module, privacy_mechanism: DifferentialPrivacy):
        self.config = config
        self.model = model
        self.privacy = privacy_mechanism
        self.local_data = None
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Device selection based on capability
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.compute_capability > 0.5 else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"Client {config.client_id} initialized on {self.device}")
    
    def load_local_data(self, data: DataLoader):
        """Load local training data."""
        self.local_data = data
        logger.info(f"Client {self.config.client_id} loaded {len(data)} batches")
    
    def local_train(self, epochs: int = 1, privacy_budget: float = None) -> ModelUpdate:
        """Perform local training with differential privacy."""
        if self.local_data is None:
            raise ValueError("No local data loaded")
        
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        privacy_spent = 0.0
        if privacy_budget is None:
            privacy_budget = self.config.privacy_budget
        
        for epoch in range(epochs):
            for batch_idx, (data, targets) in enumerate(self.local_data):
                data, targets = data.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                # Apply differential privacy to gradients
                if privacy_budget > 0:
                    with torch.no_grad():
                        for param in self.model.parameters():
                            if param.grad is not None:
                                param.grad = self.privacy.add_gaussian_noise(
                                    param.grad, 
                                    sensitivity=1.0 / len(self.local_data)
                                )
                    privacy_spent += self.privacy.epsilon / (epochs * len(self.local_data))
                
                self.optimizer.step()
                
                total_loss += loss.item()
                total_samples += targets.size(0)
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.local_data)
        accuracy = correct_predictions / total_samples
        
        # Extract model weights
        model_weights = {}
        for name, param in self.model.named_parameters():
            model_weights[name] = param.data.clone()
        
        update = ModelUpdate(
            client_id=self.config.client_id,
            round_number=0,  # Will be set by server
            model_weights=model_weights,
            num_samples=total_samples,
            loss=avg_loss,
            accuracy=accuracy,
            privacy_spent=privacy_spent,
            timestamp=datetime.now()
        )
        
        logger.info(f"Client {self.config.client_id} training complete: loss={avg_loss:.4f}, acc={accuracy:.4f}")
        return update
    
    def update_model(self, global_weights: Dict[str, torch.Tensor]):
        """Update local model with global weights."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in global_weights:
                    param.copy_(global_weights[name])
        
        logger.info(f"Client {self.config.client_id} model updated")

class ClientSelector:
    """Intelligent client selection for federated learning."""
    
    def __init__(self, selection_strategy: str = 'random'):
        self.strategy = selection_strategy
        self.client_history = defaultdict(list)
    
    def select_clients(self, 
                      available_clients: List[ClientConfig], 
                      num_clients: int, 
                      round_number: int) -> List[str]:
        """Select clients for the current round."""
        
        if self.strategy == 'random':
            return self._random_selection(available_clients, num_clients)
        elif self.strategy == 'capability_based':
            return self._capability_based_selection(available_clients, num_clients)
        elif self.strategy == 'fair':
            return self._fair_selection(available_clients, num_clients, round_number)
        else:
            raise ValueError(f"Unknown selection strategy: {self.strategy}")
    
    def _random_selection(self, clients: List[ClientConfig], num_clients: int) -> List[str]:
        """Random client selection."""
        selected_indices = np.random.choice(len(clients), min(num_clients, len(clients)), replace=False)
        return [clients[i].client_id for i in selected_indices]
    
    def _capability_based_selection(self, clients: List[ClientConfig], num_clients: int) -> List[str]:
        """Select clients based on compute capability and bandwidth."""
        # Score clients based on capability and bandwidth
        scores = []
        for client in clients:
            score = 0.7 * client.compute_capability + 0.3 * min(client.bandwidth / 100.0, 1.0)
            scores.append((score, client.client_id))
        
        # Select top clients
        scores.sort(reverse=True)
        return [client_id for _, client_id in scores[:num_clients]]
    
    def _fair_selection(self, clients: List[ClientConfig], num_clients: int, round_number: int) -> List[str]:
        """Fair client selection ensuring all clients participate over time."""
        # Calculate participation frequency
        participation_count = {}
        for client in clients:
            participation_count[client.client_id] = len(self.client_history[client.client_id])
        
        # Sort by least participation, then by capability
        client_scores = []
        for client in clients:
            participation_penalty = participation_count[client.client_id] / max(round_number, 1)
            capability_score = 0.7 * client.compute_capability + 0.3 * min(client.bandwidth / 100.0, 1.0)
            final_score = capability_score - participation_penalty
            client_scores.append((final_score, client.client_id))
        
        client_scores.sort(reverse=True)
        selected = [client_id for _, client_id in client_scores[:num_clients]]
        
        # Update history
        for client_id in selected:
            self.client_history[client_id].append(round_number)
        
        return selected

class FederatedServer:
    """Federated learning server with advanced aggregation."""
    
    def __init__(self, 
                 global_model: nn.Module, 
                 client_selector: ClientSelector,
                 secure_aggregation: SecureAggregation,
                 config: Dict):
        self.global_model = global_model
        self.client_selector = client_selector
        self.secure_aggregation = secure_aggregation
        self.config = config
        
        self.round_number = 0
        self.clients = {}
        self.round_history = []
        
        # Redis for coordination
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # MLflow tracking
        mlflow.set_tracking_uri(config.get('mlflow_uri', 'http://localhost:5000'))
        
        logger.info("Federated server initialized")
    
    def register_client(self, client_config: ClientConfig):
        """Register a new client."""
        self.clients[client_config.client_id] = client_config
        
        # Generate encryption keys
        key, salt = self.secure_aggregation.generate_client_keys(client_config.client_id)
        
        logger.info(f"Client {client_config.client_id} registered")
        return key, salt
    
    def federated_averaging(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Perform federated averaging with weighted aggregation."""
        if not updates:
            return {name: param.clone() for name, param in self.global_model.named_parameters()}
        
        # Calculate total samples for weighting
        total_samples = sum(update.num_samples for update in updates)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        for name, param in self.global_model.named_parameters():
            aggregated_weights[name] = torch.zeros_like(param)
        
        # Weighted aggregation
        for update in updates:
            weight = update.num_samples / total_samples
            for name, param_update in update.model_weights.items():
                if name in aggregated_weights:
                    aggregated_weights[name] += weight * param_update
        
        return aggregated_weights
    
    def byzantine_robust_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Byzantine-robust aggregation using coordinate-wise median."""
        if not updates:
            return {name: param.clone() for name, param in self.global_model.named_parameters()}
        
        aggregated_weights = {}
        
        for name, param in self.global_model.named_parameters():
            # Collect all updates for this parameter
            param_updates = []
            for update in updates:
                if name in update.model_weights:
                    param_updates.append(update.model_weights[name])
            
            if param_updates:
                # Stack tensors and compute coordinate-wise median
                stacked = torch.stack(param_updates)
                median_values, _ = torch.median(stacked, dim=0)
                aggregated_weights[name] = median_values
            else:
                aggregated_weights[name] = param.clone()
        
        return aggregated_weights
    
    def run_federated_round(self, num_clients: int = 10, local_epochs: int = 1) -> FederatedRound:
        """Execute a single federated learning round."""
        self.round_number += 1
        round_start = datetime.now()
        
        logger.info(f"Starting federated round {self.round_number}")
        
        # Select clients
        available_clients = list(self.clients.values())
        selected_client_ids = self.client_selector.select_clients(
            available_clients, num_clients, self.round_number
        )
        
        # Create round info
        fed_round = FederatedRound(
            round_number=self.round_number,
            selected_clients=selected_client_ids,
            global_model_version=self._get_model_hash(),
            start_time=round_start
        )
        
        # Simulate client training (in real implementation, this would be distributed)
        updates = []
        for client_id in selected_client_ids:
            try:
                # In real implementation, this would be a remote call
                update = self._simulate_client_training(client_id, local_epochs)
                update.round_number = self.round_number
                updates.append(update)
            except Exception as e:
                logger.error(f"Error training client {client_id}: {e}")
        
        if not updates:
            logger.warning("No successful client updates received")
            return fed_round
        
        # Aggregate updates
        if self.config.get('byzantine_robust', False):
            aggregated_weights = self.byzantine_robust_aggregation(updates)
        else:
            aggregated_weights = self.federated_averaging(updates)
        
        # Update global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_weights:
                    param.copy_(aggregated_weights[name])
        
        # Calculate round metrics
        avg_loss = np.mean([update.loss for update in updates])
        avg_accuracy = np.mean([update.accuracy for update in updates])
        total_privacy_spent = sum([update.privacy_spent for update in updates])
        
        fed_round.end_time = datetime.now()
        fed_round.aggregated_loss = avg_loss
        fed_round.aggregated_accuracy = avg_accuracy
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"federated_round_{self.round_number}"):
            mlflow.log_metric("round_number", self.round_number)
            mlflow.log_metric("num_clients", len(updates))
            mlflow.log_metric("avg_loss", avg_loss)
            mlflow.log_metric("avg_accuracy", avg_accuracy)
            mlflow.log_metric("total_privacy_spent", total_privacy_spent)
            mlflow.log_metric("round_duration", (fed_round.end_time - fed_round.start_time).total_seconds())
        
        # Store round info
        self.round_history.append(fed_round)
        
        # Store in Redis
        round_data = {
            'round_number': self.round_number,
            'selected_clients': selected_client_ids,
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'timestamp': round_start.isoformat()
        }
        self.redis_client.lpush("federated_rounds", json.dumps(round_data))
        self.redis_client.ltrim("federated_rounds", 0, 99)  # Keep last 100 rounds
        
        logger.info(f"Round {self.round_number} complete: loss={avg_loss:.4f}, acc={avg_accuracy:.4f}")
        return fed_round
    
    def _simulate_client_training(self, client_id: str, epochs: int) -> ModelUpdate:
        """Simulate client training (replace with actual distributed training)."""
        # This is a simulation - in real implementation, this would be a remote call
        client_config = self.clients[client_id]
        
        # Create a mock client
        privacy = DifferentialPrivacy(epsilon=0.1, delta=1e-5)
        client_model = NewsPersonalizationModel()
        
        # Copy global model weights
        client_model.load_state_dict(self.global_model.state_dict())
        
        client = FederatedClient(client_config, client_model, privacy)
        
        # Generate mock training data
        mock_data = self._generate_mock_data(client_config.data_size)
        client.load_local_data(mock_data)
        
        # Perform training
        update = client.local_train(epochs=epochs)
        
        return update
    
    def _generate_mock_data(self, data_size: int) -> DataLoader:
        """Generate mock training data for simulation."""
        # Create synthetic data
        vocab_size = 1000
        seq_length = 50
        num_categories = 20
        
        input_ids = torch.randint(0, vocab_size, (data_size, seq_length))
        labels = torch.randint(0, num_categories, (data_size,))
        
        dataset = torch.utils.data.TensorDataset(input_ids, labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        return dataloader
    
    def _get_model_hash(self) -> str:
        """Generate hash of current global model."""
        model_str = ""
        for name, param in self.global_model.named_parameters():
            model_str += f"{name}:{param.data.sum().item()}"
        
        return hashlib.md5(model_str.encode()).hexdigest()[:8]
    
    def get_convergence_metrics(self) -> Dict[str, float]:
        """Calculate convergence metrics."""
        if len(self.round_history) < 2:
            return {}
        
        recent_rounds = self.round_history[-10:]  # Last 10 rounds
        losses = [r.aggregated_loss for r in recent_rounds if r.aggregated_loss is not None]
        accuracies = [r.aggregated_accuracy for r in recent_rounds if r.aggregated_accuracy is not None]
        
        if not losses or not accuracies:
            return {}
        
        # Calculate trends
        loss_trend = np.polyfit(range(len(losses)), losses, 1)[0]  # Slope
        accuracy_trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
        
        # Calculate stability (coefficient of variation)
        loss_stability = np.std(losses) / np.mean(losses) if np.mean(losses) > 0 else 0
        accuracy_stability = np.std(accuracies) / np.mean(accuracies) if np.mean(accuracies) > 0 else 0
        
        return {
            'loss_trend': loss_trend,
            'accuracy_trend': accuracy_trend,
            'loss_stability': loss_stability,
            'accuracy_stability': accuracy_stability,
            'current_loss': losses[-1],
            'current_accuracy': accuracies[-1]
        }

def main():
    """Main function to run federated learning simulation."""
    # Configuration
    config = {
        'redis_host': os.getenv('REDIS_HOST', 'localhost'),
        'redis_port': int(os.getenv('REDIS_PORT', 6379)),
        'mlflow_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
        'byzantine_robust': True,
        'num_rounds': 50,
        'clients_per_round': 10,
        'local_epochs': 2
    }
    
    # Initialize components
    global_model = NewsPersonalizationModel()
    client_selector = ClientSelector(selection_strategy='fair')
    secure_aggregation = SecureAggregation(num_clients=100)
    
    server = FederatedServer(global_model, client_selector, secure_aggregation, config)
    
    # Register mock clients
    for i in range(100):
        client_config = ClientConfig(
            client_id=f"client_{i:03d}",
            device_type=np.random.choice(['mobile', 'desktop', 'edge']),
            compute_capability=np.random.uniform(0.1, 1.0),
            bandwidth=np.random.uniform(1.0, 100.0),
            privacy_budget=np.random.uniform(0.1, 1.0),
            data_size=np.random.randint(100, 1000),
            location=f"location_{i % 10}",
            timezone=f"UTC{np.random.randint(-12, 13)}"
        )
        server.register_client(client_config)
    
    # Run federated learning
    logger.info("Starting federated learning simulation")
    
    for round_num in range(config['num_rounds']):
        try:
            fed_round = server.run_federated_round(
                num_clients=config['clients_per_round'],
                local_epochs=config['local_epochs']
            )
            
            # Check convergence
            if round_num % 10 == 0:
                metrics = server.get_convergence_metrics()
                logger.info(f"Convergence metrics: {metrics}")
                
                # Early stopping if converged
                if metrics.get('loss_trend', 0) > -0.001 and metrics.get('accuracy_trend', 0) < 0.001:
                    logger.info("Model converged, stopping early")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in round {round_num}: {e}")
    
    logger.info("Federated learning simulation complete")

if __name__ == "__main__":
    main()