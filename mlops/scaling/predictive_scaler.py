#!/usr/bin/env python3
"""
Predictive Scaling Module - AI-Driven Auto-Scaling for Deployments

This module implements predictive scaling using machine learning to forecast
resource demands and proactively scale applications before traffic spikes.

Features:
- Time series forecasting with Prophet and LSTM
- Multi-metric scaling decisions (CPU, memory, RPS, latency)
- Integration with Kubernetes HPA and VPA
- Cost optimization and SLA adherence
- Anomaly detection for scaling events
- Multi-cloud scaling coordination

Author: Commander Solaris "DeployX" Vivante
Version: 1.0.0
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import yaml
import requests

logger = logging.getLogger(__name__)

class ScalingDirection(Enum):
    """Scaling direction"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

class ScalingTrigger(Enum):
    """Scaling trigger types"""
    PREDICTIVE = "predictive"
    REACTIVE = "reactive"
    SCHEDULED = "scheduled"
    MANUAL = "manual"

class ResourceType(Enum):
    """Resource types for scaling"""
    CPU = "cpu"
    MEMORY = "memory"
    REPLICAS = "replicas"
    STORAGE = "storage"

@dataclass
class ScalingMetric:
    """Scaling metric definition"""
    name: str
    current_value: float
    predicted_value: float
    threshold_up: float
    threshold_down: float
    weight: float
    unit: str

@dataclass
class ScalingDecision:
    """Scaling decision"""
    timestamp: datetime
    direction: ScalingDirection
    trigger: ScalingTrigger
    resource_type: ResourceType
    current_value: int
    target_value: int
    confidence: float
    reasoning: str
    metrics: List[ScalingMetric]
    cost_impact: float
    sla_impact: float

@dataclass
class ScalingEvent:
    """Scaling event record"""
    id: str
    decision: ScalingDecision
    execution_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    success: bool = False
    error: Optional[str] = None
    actual_impact: Optional[Dict[str, float]] = None

class PredictiveScaler:
    """
    AI-driven predictive scaling system
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.models = {}
        self.scalers = {}
        self.scaling_history: List[ScalingEvent] = []
        self.prometheus_url = self.config.get("prometheus_url", "http://localhost:9090")
        self.kubernetes_enabled = self.config.get("kubernetes_enabled", False)
        self.anomaly_detector = None
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default predictive scaling configuration"""
        return {
            "prometheus_url": "http://localhost:9090",
            "kubernetes_enabled": False,
            "prediction_horizon": 300,  # 5 minutes ahead
            "scaling_cooldown": 180,    # 3 minutes between scaling events
            "metrics": {
                "cpu_utilization": {
                    "enabled": True,
                    "threshold_up": 70.0,
                    "threshold_down": 30.0,
                    "weight": 0.4,
                    "query": "avg(rate(cpu_usage_seconds_total[5m])) * 100"
                },
                "memory_utilization": {
                    "enabled": True,
                    "threshold_up": 80.0,
                    "threshold_down": 40.0,
                    "weight": 0.3,
                    "query": "avg(memory_usage_bytes / memory_limit_bytes) * 100"
                },
                "request_rate": {
                    "enabled": True,
                    "threshold_up": 1000.0,
                    "threshold_down": 200.0,
                    "weight": 0.2,
                    "query": "sum(rate(http_requests_total[5m]))"
                },
                "response_time": {
                    "enabled": True,
                    "threshold_up": 500.0,  # 500ms
                    "threshold_down": 100.0,
                    "weight": 0.1,
                    "query": "avg(http_request_duration_seconds) * 1000"
                }
            },
            "scaling_limits": {
                "min_replicas": 2,
                "max_replicas": 50,
                "min_cpu": 100,      # 100m
                "max_cpu": 4000,     # 4 cores
                "min_memory": 128,   # 128Mi
                "max_memory": 8192   # 8Gi
            },
            "cost_optimization": {
                "enabled": True,
                "cost_per_replica_hour": 0.05,
                "sla_penalty_per_minute": 10.0,
                "max_cost_increase": 0.2  # 20%
            },
            "ml_models": {
                "prophet": {
                    "enabled": True,
                    "seasonality_mode": "multiplicative",
                    "yearly_seasonality": False,
                    "weekly_seasonality": True,
                    "daily_seasonality": True
                },
                "lstm": {
                    "enabled": True,
                    "sequence_length": 60,
                    "hidden_units": 50,
                    "epochs": 100
                }
            }
        }
    
    async def initialize(self):
        """Initialize the Predictive Scaler"""
        logger.info("Initializing Predictive Scaler...")
        
        try:
            # Test monitoring connectivity
            await self._test_monitoring_connection()
            
            # Initialize ML models
            await self._initialize_models()
            
            # Initialize anomaly detector
            self._initialize_anomaly_detector()
            
            # Load historical data
            await self._load_historical_data()
            
            logger.info("Predictive Scaler initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Predictive Scaler: {e}")
            raise
    
    async def _test_monitoring_connection(self):
        """Test connection to monitoring systems"""
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/query", 
                                  params={"query": "up"}, timeout=10)
            response.raise_for_status()
            logger.info("Monitoring connection successful")
        except Exception as e:
            logger.warning(f"Monitoring connection failed: {e}")
    
    async def _initialize_models(self):
        """Initialize ML models for prediction"""
        logger.info("Initializing ML models...")
        
        # Initialize Prophet models for each metric
        if self.config["ml_models"]["prophet"]["enabled"]:
            try:
                from prophet import Prophet
                
                for metric_name in self.config["metrics"]:
                    if self.config["metrics"][metric_name]["enabled"]:
                        model = Prophet(
                            seasonality_mode=self.config["ml_models"]["prophet"]["seasonality_mode"],
                            yearly_seasonality=self.config["ml_models"]["prophet"]["yearly_seasonality"],
                            weekly_seasonality=self.config["ml_models"]["prophet"]["weekly_seasonality"],
                            daily_seasonality=self.config["ml_models"]["prophet"]["daily_seasonality"]
                        )
                        self.models[f"prophet_{metric_name}"] = model
                        
                logger.info("Prophet models initialized")
            except ImportError:
                logger.warning("Prophet not available, using fallback prediction")
        
        # Initialize LSTM models (simplified version)
        if self.config["ml_models"]["lstm"]["enabled"]:
            try:
                # In a real implementation, you would use TensorFlow/PyTorch
                # For demo, we'll use a simple linear model as placeholder
                from sklearn.linear_model import LinearRegression
                
                for metric_name in self.config["metrics"]:
                    if self.config["metrics"][metric_name]["enabled"]:
                        model = LinearRegression()
                        self.models[f"lstm_{metric_name}"] = model
                        
                        # Initialize scaler for this metric
                        scaler = StandardScaler()
                        self.scalers[f"lstm_{metric_name}"] = scaler
                        
                logger.info("LSTM models initialized")
            except Exception as e:
                logger.warning(f"LSTM model initialization failed: {e}")
    
    def _initialize_anomaly_detector(self):
        """Initialize anomaly detection for scaling events"""
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        logger.info("Anomaly detector initialized")
    
    async def _load_historical_data(self):
        """Load historical metrics data for model training"""
        logger.info("Loading historical data...")
        
        try:
            # Load data for the past 7 days
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)
            
            historical_data = {}
            
            for metric_name, metric_config in self.config["metrics"].items():
                if metric_config["enabled"]:
                    data = await self._fetch_historical_metric_data(
                        metric_name, metric_config["query"], start_time, end_time
                    )
                    historical_data[metric_name] = data
            
            # Train models with historical data
            await self._train_models(historical_data)
            
            logger.info("Historical data loaded and models trained")
            
        except Exception as e:
            logger.warning(f"Failed to load historical data: {e}")
            # Generate synthetic data for demo
            await self._generate_synthetic_training_data()
    
    async def _fetch_historical_metric_data(self, metric_name: str, query: str, 
                                          start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch historical metric data from Prometheus"""
        try:
            # Convert to Unix timestamps
            start_ts = int(start_time.timestamp())
            end_ts = int(end_time.timestamp())
            step = 300  # 5-minute intervals
            
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query_range",
                params={
                    "query": query,
                    "start": start_ts,
                    "end": end_ts,
                    "step": step
                },
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data["status"] == "success" and data["data"]["result"]:
                # Parse Prometheus response
                values = data["data"]["result"][0]["values"]
                df = pd.DataFrame(values, columns=["timestamp", "value"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                df["value"] = pd.to_numeric(df["value"])
                df.set_index("timestamp", inplace=True)
                
                logger.info(f"Fetched {len(df)} data points for {metric_name}")
                return df
            else:
                logger.warning(f"No data returned for metric {metric_name}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to fetch data for {metric_name}: {e}")
            return pd.DataFrame()
    
    async def _generate_synthetic_training_data(self):
        """Generate synthetic training data for demo purposes"""
        logger.info("Generating synthetic training data...")
        
        # Generate 7 days of synthetic data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        timestamps = pd.date_range(start=start_time, end=end_time, freq="5T")
        
        synthetic_data = {}
        
        for metric_name in self.config["metrics"]:
            if self.config["metrics"][metric_name]["enabled"]:
                # Generate realistic synthetic data with patterns
                base_values = self._generate_metric_pattern(metric_name, len(timestamps))
                
                df = pd.DataFrame({
                    "value": base_values
                }, index=timestamps)
                
                synthetic_data[metric_name] = df
        
        # Train models with synthetic data
        await self._train_models(synthetic_data)
        
        logger.info("Synthetic training data generated and models trained")
    
    def _generate_metric_pattern(self, metric_name: str, num_points: int) -> np.ndarray:
        """Generate realistic metric patterns"""
        t = np.linspace(0, 7 * 24, num_points)  # 7 days in hours
        
        if metric_name == "cpu_utilization":
            # CPU with daily and weekly patterns
            base = 40
            daily_pattern = 20 * np.sin(2 * np.pi * t / 24)  # Daily cycle
            weekly_pattern = 10 * np.sin(2 * np.pi * t / (24 * 7))  # Weekly cycle
            noise = np.random.normal(0, 5, num_points)
            values = base + daily_pattern + weekly_pattern + noise
            return np.clip(values, 0, 100)
            
        elif metric_name == "memory_utilization":
            # Memory with gradual increase and daily patterns
            base = 50
            trend = 0.1 * t  # Gradual increase
            daily_pattern = 15 * np.sin(2 * np.pi * t / 24)
            noise = np.random.normal(0, 3, num_points)
            values = base + trend + daily_pattern + noise
            return np.clip(values, 0, 100)
            
        elif metric_name == "request_rate":
            # Request rate with business hours pattern
            base = 500
            business_hours = 300 * (np.sin(2 * np.pi * (t % 24 - 6) / 12) > 0) * np.sin(2 * np.pi * (t % 24 - 6) / 12)
            weekly_pattern = 200 * (1 - 0.5 * np.sin(2 * np.pi * t / (24 * 7)))
            noise = np.random.normal(0, 50, num_points)
            values = base + business_hours + weekly_pattern + noise
            return np.clip(values, 0, None)
            
        elif metric_name == "response_time":
            # Response time inversely correlated with load
            base = 200
            load_effect = 100 * np.sin(2 * np.pi * t / 24)  # Higher during peak hours
            noise = np.random.normal(0, 20, num_points)
            values = base + load_effect + noise
            return np.clip(values, 50, 2000)
            
        else:
            # Default pattern
            base = 50
            pattern = 20 * np.sin(2 * np.pi * t / 24)
            noise = np.random.normal(0, 5, num_points)
            return base + pattern + noise
    
    async def _train_models(self, historical_data: Dict[str, pd.DataFrame]):
        """Train ML models with historical data"""
        logger.info("Training ML models...")
        
        for metric_name, df in historical_data.items():
            if df.empty:
                continue
                
            try:
                # Train Prophet model
                if f"prophet_{metric_name}" in self.models:
                    await self._train_prophet_model(metric_name, df)
                
                # Train LSTM model (simplified)
                if f"lstm_{metric_name}" in self.models:
                    await self._train_lstm_model(metric_name, df)
                    
            except Exception as e:
                logger.error(f"Failed to train models for {metric_name}: {e}")
        
        # Train anomaly detector with scaling history
        if self.scaling_history:
            self._train_anomaly_detector()
        
        logger.info("Model training completed")
    
    async def _train_prophet_model(self, metric_name: str, df: pd.DataFrame):
        """Train Prophet model for a specific metric"""
        try:
            model = self.models[f"prophet_{metric_name}"]
            
            # Prepare data for Prophet
            prophet_df = df.reset_index()
            prophet_df.columns = ["ds", "y"]
            
            # Train model
            model.fit(prophet_df)
            
            logger.info(f"Prophet model trained for {metric_name}")
            
        except Exception as e:
            logger.error(f"Prophet training failed for {metric_name}: {e}")
    
    async def _train_lstm_model(self, metric_name: str, df: pd.DataFrame):
        """Train LSTM model for a specific metric"""
        try:
            model = self.models[f"lstm_{metric_name}"]
            scaler = self.scalers[f"lstm_{metric_name}"]
            
            # Prepare data
            values = df["value"].values.reshape(-1, 1)
            scaled_values = scaler.fit_transform(values)
            
            # Create sequences (simplified approach)
            sequence_length = self.config["ml_models"]["lstm"]["sequence_length"]
            X, y = [], []
            
            for i in range(sequence_length, len(scaled_values)):
                X.append(scaled_values[i-sequence_length:i].flatten())
                y.append(scaled_values[i][0])
            
            if X and y:
                X = np.array(X)
                y = np.array(y)
                
                # Train simple linear model as LSTM placeholder
                model.fit(X, y)
                
                logger.info(f"LSTM model trained for {metric_name}")
            
        except Exception as e:
            logger.error(f"LSTM training failed for {metric_name}: {e}")
    
    def _train_anomaly_detector(self):
        """Train anomaly detector with scaling history"""
        try:
            if len(self.scaling_history) < 10:
                return  # Need more data
            
            # Extract features from scaling events
            features = []
            for event in self.scaling_history:
                feature_vector = [
                    event.decision.current_value,
                    event.decision.target_value,
                    event.decision.confidence,
                    event.decision.cost_impact,
                    event.decision.sla_impact,
                    len(event.decision.metrics)
                ]
                features.append(feature_vector)
            
            features = np.array(features)
            self.anomaly_detector.fit(features)
            
            logger.info("Anomaly detector trained")
            
        except Exception as e:
            logger.error(f"Anomaly detector training failed: {e}")
    
    async def predict_metrics(self, horizon_minutes: int = 5) -> Dict[str, float]:
        """Predict metric values for the specified horizon"""
        predictions = {}
        
        for metric_name in self.config["metrics"]:
            if self.config["metrics"][metric_name]["enabled"]:
                try:
                    # Get current metric value
                    current_value = await self._get_current_metric_value(metric_name)
                    
                    # Predict using available models
                    prophet_pred = await self._predict_with_prophet(metric_name, horizon_minutes)
                    lstm_pred = await self._predict_with_lstm(metric_name, horizon_minutes)
                    
                    # Ensemble prediction
                    if prophet_pred is not None and lstm_pred is not None:
                        prediction = 0.6 * prophet_pred + 0.4 * lstm_pred
                    elif prophet_pred is not None:
                        prediction = prophet_pred
                    elif lstm_pred is not None:
                        prediction = lstm_pred
                    else:
                        # Fallback to trend-based prediction
                        prediction = await self._predict_with_trend(metric_name, current_value)
                    
                    predictions[metric_name] = prediction
                    
                except Exception as e:
                    logger.error(f"Prediction failed for {metric_name}: {e}")
                    # Use current value as fallback
                    predictions[metric_name] = await self._get_current_metric_value(metric_name)
        
        logger.info(f"Generated predictions for {len(predictions)} metrics")
        return predictions
    
    async def _get_current_metric_value(self, metric_name: str) -> float:
        """Get current value of a metric"""
        try:
            query = self.config["metrics"][metric_name]["query"]
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            if data["status"] == "success" and data["data"]["result"]:
                value = float(data["data"]["result"][0]["value"][1])
                return value
            else:
                logger.warning(f"No current data for {metric_name}")
                return self._get_fallback_metric_value(metric_name)
                
        except Exception as e:
            logger.error(f"Failed to get current value for {metric_name}: {e}")
            return self._get_fallback_metric_value(metric_name)
    
    def _get_fallback_metric_value(self, metric_name: str) -> float:
        """Get fallback metric value for demo"""
        fallback_values = {
            "cpu_utilization": 45.0,
            "memory_utilization": 60.0,
            "request_rate": 500.0,
            "response_time": 200.0
        }
        return fallback_values.get(metric_name, 50.0)
    
    async def _predict_with_prophet(self, metric_name: str, horizon_minutes: int) -> Optional[float]:
        """Predict using Prophet model"""
        try:
            model_key = f"prophet_{metric_name}"
            if model_key not in self.models:
                return None
            
            model = self.models[model_key]
            
            # Create future dataframe
            future_time = datetime.now() + timedelta(minutes=horizon_minutes)
            future_df = pd.DataFrame({"ds": [future_time]})
            
            # Make prediction
            forecast = model.predict(future_df)
            prediction = forecast["yhat"].iloc[0]
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Prophet prediction failed for {metric_name}: {e}")
            return None
    
    async def _predict_with_lstm(self, metric_name: str, horizon_minutes: int) -> Optional[float]:
        """Predict using LSTM model"""
        try:
            model_key = f"lstm_{metric_name}"
            if model_key not in self.models or model_key not in self.scalers:
                return None
            
            model = self.models[model_key]
            scaler = self.scalers[model_key]
            
            # Get recent data for sequence
            sequence_length = self.config["ml_models"]["lstm"]["sequence_length"]
            
            # For demo, generate a simple sequence
            current_value = await self._get_current_metric_value(metric_name)
            sequence = np.array([current_value] * sequence_length).reshape(1, -1)
            
            # Scale and predict
            scaled_sequence = scaler.transform(sequence.reshape(-1, 1)).flatten().reshape(1, -1)
            scaled_prediction = model.predict(scaled_sequence)[0]
            
            # Inverse transform
            prediction = scaler.inverse_transform([[scaled_prediction]])[0][0]
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"LSTM prediction failed for {metric_name}: {e}")
            return None
    
    async def _predict_with_trend(self, metric_name: str, current_value: float) -> float:
        """Simple trend-based prediction as fallback"""
        # Simple trend based on time of day
        current_hour = datetime.now().hour
        
        if metric_name == "cpu_utilization":
            # Higher during business hours
            if 9 <= current_hour <= 17:
                return current_value * 1.2
            else:
                return current_value * 0.9
        elif metric_name == "request_rate":
            # Peak during business hours
            if 9 <= current_hour <= 17:
                return current_value * 1.5
            else:
                return current_value * 0.7
        else:
            # Default: slight increase
            return current_value * 1.1
    
    async def make_scaling_decision(self, target: str) -> Optional[ScalingDecision]:
        """Make a scaling decision based on predictions"""
        logger.info(f"Making scaling decision for target: {target}")
        
        try:
            # Get current metrics
            current_metrics = await self._collect_current_metrics()
            
            # Get predictions
            predicted_metrics = await self.predict_metrics(
                self.config["prediction_horizon"] // 60
            )
            
            # Create scaling metrics
            scaling_metrics = []
            for metric_name in self.config["metrics"]:
                if self.config["metrics"][metric_name]["enabled"]:
                    metric_config = self.config["metrics"][metric_name]
                    
                    scaling_metric = ScalingMetric(
                        name=metric_name,
                        current_value=current_metrics.get(metric_name, 0),
                        predicted_value=predicted_metrics.get(metric_name, 0),
                        threshold_up=metric_config["threshold_up"],
                        threshold_down=metric_config["threshold_down"],
                        weight=metric_config["weight"],
                        unit=self._get_metric_unit(metric_name)
                    )
                    scaling_metrics.append(scaling_metric)
            
            # Analyze scaling need
            scaling_analysis = self._analyze_scaling_need(scaling_metrics)
            
            if scaling_analysis["direction"] == ScalingDirection.STABLE:
                logger.info("No scaling needed")
                return None
            
            # Get current resource values
            current_replicas = await self._get_current_replicas(target)
            
            # Calculate target scaling
            target_replicas = self._calculate_target_replicas(
                current_replicas, scaling_analysis, scaling_metrics
            )
            
            # Validate scaling limits
            target_replicas = self._apply_scaling_limits(target_replicas)
            
            if target_replicas == current_replicas:
                logger.info("Target replicas same as current, no scaling needed")
                return None
            
            # Calculate impact
            cost_impact = self._calculate_cost_impact(current_replicas, target_replicas)
            sla_impact = self._calculate_sla_impact(scaling_metrics)
            
            # Create scaling decision
            decision = ScalingDecision(
                timestamp=datetime.now(),
                direction=scaling_analysis["direction"],
                trigger=ScalingTrigger.PREDICTIVE,
                resource_type=ResourceType.REPLICAS,
                current_value=current_replicas,
                target_value=target_replicas,
                confidence=scaling_analysis["confidence"],
                reasoning=scaling_analysis["reasoning"],
                metrics=scaling_metrics,
                cost_impact=cost_impact,
                sla_impact=sla_impact
            )
            
            # Check for anomalies
            if self._is_scaling_anomalous(decision):
                logger.warning("Scaling decision flagged as anomalous, requiring review")
                decision.confidence *= 0.5  # Reduce confidence
            
            logger.info(f"Scaling decision: {decision.direction.value} from {current_replicas} to {target_replicas} replicas")
            return decision
            
        except Exception as e:
            logger.error(f"Failed to make scaling decision: {e}")
            return None
    
    async def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current metric values"""
        metrics = {}
        
        for metric_name in self.config["metrics"]:
            if self.config["metrics"][metric_name]["enabled"]:
                metrics[metric_name] = await self._get_current_metric_value(metric_name)
        
        return metrics
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get unit for a metric"""
        units = {
            "cpu_utilization": "%",
            "memory_utilization": "%",
            "request_rate": "RPS",
            "response_time": "ms"
        }
        return units.get(metric_name, "")
    
    def _analyze_scaling_need(self, metrics: List[ScalingMetric]) -> Dict[str, Any]:
        """Analyze if scaling is needed based on metrics"""
        scale_up_score = 0.0
        scale_down_score = 0.0
        reasoning_parts = []
        
        for metric in metrics:
            # Check if predicted value exceeds thresholds
            if metric.predicted_value > metric.threshold_up:
                score = ((metric.predicted_value - metric.threshold_up) / metric.threshold_up) * metric.weight
                scale_up_score += score
                reasoning_parts.append(f"{metric.name} predicted to exceed threshold ({metric.predicted_value:.1f} > {metric.threshold_up})")
            
            elif metric.predicted_value < metric.threshold_down:
                score = ((metric.threshold_down - metric.predicted_value) / metric.threshold_down) * metric.weight
                scale_down_score += score
                reasoning_parts.append(f"{metric.name} predicted below threshold ({metric.predicted_value:.1f} < {metric.threshold_down})")
        
        # Determine scaling direction
        if scale_up_score > scale_down_score and scale_up_score > 0.3:
            direction = ScalingDirection.UP
            confidence = min(0.95, scale_up_score)
        elif scale_down_score > scale_up_score and scale_down_score > 0.3:
            direction = ScalingDirection.DOWN
            confidence = min(0.95, scale_down_score)
        else:
            direction = ScalingDirection.STABLE
            confidence = 0.8
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "All metrics within normal ranges"
        
        return {
            "direction": direction,
            "confidence": confidence,
            "scale_up_score": scale_up_score,
            "scale_down_score": scale_down_score,
            "reasoning": reasoning
        }
    
    async def _get_current_replicas(self, target: str) -> int:
        """Get current number of replicas for target"""
        try:
            if self.kubernetes_enabled:
                # In real implementation, query Kubernetes API
                # For demo, return a simulated value
                pass
            
            # Fallback: return simulated current replicas
            return 3
            
        except Exception as e:
            logger.error(f"Failed to get current replicas for {target}: {e}")
            return 3
    
    def _calculate_target_replicas(self, current_replicas: int, 
                                 scaling_analysis: Dict[str, Any], 
                                 metrics: List[ScalingMetric]) -> int:
        """Calculate target number of replicas"""
        if scaling_analysis["direction"] == ScalingDirection.UP:
            # Scale up based on the highest metric pressure
            max_scale_factor = 1.0
            
            for metric in metrics:
                if metric.predicted_value > metric.threshold_up:
                    scale_factor = metric.predicted_value / metric.threshold_up
                    max_scale_factor = max(max_scale_factor, scale_factor)
            
            target = int(current_replicas * max_scale_factor)
            return min(target, current_replicas + 5)  # Limit scale-up rate
            
        elif scaling_analysis["direction"] == ScalingDirection.DOWN:
            # Scale down conservatively
            min_scale_factor = 1.0
            
            for metric in metrics:
                if metric.predicted_value < metric.threshold_down:
                    scale_factor = metric.predicted_value / metric.threshold_down
                    min_scale_factor = min(min_scale_factor, scale_factor)
            
            target = int(current_replicas * min_scale_factor)
            return max(target, current_replicas - 2)  # Limit scale-down rate
        
        return current_replicas
    
    def _apply_scaling_limits(self, target_replicas: int) -> int:
        """Apply scaling limits to target replicas"""
        limits = self.config["scaling_limits"]
        return max(limits["min_replicas"], min(limits["max_replicas"], target_replicas))
    
    def _calculate_cost_impact(self, current_replicas: int, target_replicas: int) -> float:
        """Calculate cost impact of scaling decision"""
        cost_config = self.config["cost_optimization"]
        
        if not cost_config["enabled"]:
            return 0.0
        
        replica_diff = target_replicas - current_replicas
        hourly_cost_change = replica_diff * cost_config["cost_per_replica_hour"]
        
        # Return percentage change
        if current_replicas > 0:
            return (hourly_cost_change / (current_replicas * cost_config["cost_per_replica_hour"])) * 100
        else:
            return 0.0
    
    def _calculate_sla_impact(self, metrics: List[ScalingMetric]) -> float:
        """Calculate SLA impact based on metrics"""
        sla_risk = 0.0
        
        for metric in metrics:
            if metric.name == "response_time" and metric.predicted_value > metric.threshold_up:
                # High response time increases SLA risk
                sla_risk += (metric.predicted_value - metric.threshold_up) / metric.threshold_up
            elif metric.name == "error_rate" and metric.predicted_value > metric.threshold_up:
                # High error rate significantly increases SLA risk
                sla_risk += 2 * (metric.predicted_value - metric.threshold_up) / metric.threshold_up
        
        return min(100.0, sla_risk * 100)  # Return as percentage
    
    def _is_scaling_anomalous(self, decision: ScalingDecision) -> bool:
        """Check if scaling decision is anomalous"""
        if self.anomaly_detector is None or len(self.scaling_history) < 10:
            return False
        
        try:
            # Extract features from decision
            features = np.array([[
                decision.current_value,
                decision.target_value,
                decision.confidence,
                decision.cost_impact,
                decision.sla_impact,
                len(decision.metrics)
            ]])
            
            # Predict anomaly
            anomaly_score = self.anomaly_detector.decision_function(features)[0]
            is_anomaly = self.anomaly_detector.predict(features)[0] == -1
            
            if is_anomaly:
                logger.warning(f"Anomalous scaling decision detected (score: {anomaly_score:.3f})")
            
            return is_anomaly
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return False
    
    async def execute_scaling_decision(self, decision: ScalingDecision, target: str) -> ScalingEvent:
        """Execute a scaling decision"""
        event_id = f"scale-{int(time.time())}-{decision.direction.value}"
        
        event = ScalingEvent(
            id=event_id,
            decision=decision,
            execution_time=datetime.now()
        )
        
        logger.info(f"Executing scaling decision {event_id}: {decision.direction.value} to {decision.target_value} replicas")
        
        try:
            # Execute scaling based on resource type
            if decision.resource_type == ResourceType.REPLICAS:
                success = await self._scale_replicas(target, decision.target_value)
            else:
                success = await self._scale_resources(target, decision.resource_type, decision.target_value)
            
            event.success = success
            event.completion_time = datetime.now()
            
            if success:
                logger.info(f"Scaling executed successfully: {event_id}")
                
                # Wait and collect impact metrics
                await asyncio.sleep(60)  # Wait 1 minute
                event.actual_impact = await self._measure_scaling_impact(decision)
            else:
                event.error = "Scaling execution failed"
                logger.error(f"Scaling execution failed: {event_id}")
            
        except Exception as e:
            event.success = False
            event.error = str(e)
            event.completion_time = datetime.now()
            logger.error(f"Scaling execution error for {event_id}: {e}")
        
        # Store event in history
        self.scaling_history.append(event)
        
        # Retrain anomaly detector periodically
        if len(self.scaling_history) % 10 == 0:
            self._train_anomaly_detector()
        
        return event
    
    async def _scale_replicas(self, target: str, target_replicas: int) -> bool:
        """Scale replicas for target deployment"""
        try:
            if self.kubernetes_enabled:
                # In real implementation, use Kubernetes API
                # kubectl scale deployment {target} --replicas={target_replicas}
                pass
            
            # For demo, simulate scaling
            logger.info(f"Simulating replica scaling for {target} to {target_replicas} replicas")
            await asyncio.sleep(2)  # Simulate scaling time
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale replicas for {target}: {e}")
            return False
    
    async def _scale_resources(self, target: str, resource_type: ResourceType, target_value: int) -> bool:
        """Scale resources (CPU/memory) for target deployment"""
        try:
            if self.kubernetes_enabled:
                # In real implementation, update resource requests/limits
                pass
            
            # For demo, simulate resource scaling
            logger.info(f"Simulating {resource_type.value} scaling for {target} to {target_value}")
            await asyncio.sleep(2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale {resource_type.value} for {target}: {e}")
            return False
    
    async def _measure_scaling_impact(self, decision: ScalingDecision) -> Dict[str, float]:
        """Measure actual impact of scaling decision"""
        try:
            # Collect metrics after scaling
            post_scaling_metrics = await self._collect_current_metrics()
            
            # Calculate impact
            impact = {}
            for metric in decision.metrics:
                if metric.name in post_scaling_metrics:
                    actual_value = post_scaling_metrics[metric.name]
                    predicted_value = metric.predicted_value
                    
                    impact[f"{metric.name}_actual"] = actual_value
                    impact[f"{metric.name}_predicted"] = predicted_value
                    impact[f"{metric.name}_accuracy"] = 1.0 - abs(actual_value - predicted_value) / max(actual_value, predicted_value)
            
            logger.info(f"Scaling impact measured: {impact}")
            return impact
            
        except Exception as e:
            logger.error(f"Failed to measure scaling impact: {e}")
            return {}
    
    def get_scaling_recommendations(self, target: str) -> Dict[str, Any]:
        """Get scaling recommendations for a target"""
        recommendations = {
            "target": target,
            "timestamp": datetime.now().isoformat(),
            "recommendations": [],
            "current_status": {},
            "optimization_opportunities": []
        }
        
        try:
            # Analyze recent scaling history
            recent_events = [e for e in self.scaling_history 
                           if e.execution_time and 
                           (datetime.now() - e.execution_time).days < 7]
            
            if recent_events:
                # Calculate scaling frequency
                scaling_frequency = len(recent_events) / 7  # per day
                
                if scaling_frequency > 5:
                    recommendations["recommendations"].append(
                        "High scaling frequency detected - consider adjusting thresholds or improving prediction accuracy"
                    )
                
                # Analyze scaling accuracy
                accurate_predictions = sum(1 for e in recent_events 
                                         if e.actual_impact and 
                                         any(v > 0.8 for k, v in e.actual_impact.items() if k.endswith('_accuracy')))
                
                if recent_events and accurate_predictions / len(recent_events) < 0.7:
                    recommendations["recommendations"].append(
                        "Low prediction accuracy - consider retraining models with more recent data"
                    )
            
            # Cost optimization opportunities
            if self.config["cost_optimization"]["enabled"]:
                recommendations["optimization_opportunities"].append(
                    "Enable scheduled scaling for predictable workloads to reduce costs"
                )
            
            # Performance optimization
            recommendations["optimization_opportunities"].extend([
                "Consider implementing custom metrics for more accurate scaling decisions",
                "Evaluate using Vertical Pod Autoscaler (VPA) in addition to HPA",
                "Implement predictive scaling for known traffic patterns"
            ])
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            recommendations["error"] = str(e)
        
        return recommendations
    
    def export_scaling_report(self, days: int = 7) -> Dict[str, Any]:
        """Export scaling report for analysis"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Filter events in time range
        events_in_range = [
            e for e in self.scaling_history
            if e.execution_time and start_time <= e.execution_time <= end_time
        ]
        
        report = {
            "report_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "days": days
            },
            "summary": {
                "total_scaling_events": len(events_in_range),
                "successful_events": sum(1 for e in events_in_range if e.success),
                "failed_events": sum(1 for e in events_in_range if not e.success),
                "scale_up_events": sum(1 for e in events_in_range if e.decision.direction == ScalingDirection.UP),
                "scale_down_events": sum(1 for e in events_in_range if e.decision.direction == ScalingDirection.DOWN)
            },
            "events": [asdict(event) for event in events_in_range],
            "generated_at": datetime.now().isoformat()
        }
        
        # Calculate metrics
        if events_in_range:
            avg_confidence = sum(e.decision.confidence for e in events_in_range) / len(events_in_range)
            report["summary"]["average_confidence"] = avg_confidence
            
            # Cost impact
            total_cost_impact = sum(e.decision.cost_impact for e in events_in_range)
            report["summary"]["total_cost_impact_percent"] = total_cost_impact
        
        return report

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_predictive_scaler():
        """Test the Predictive Scaler"""
        scaler = PredictiveScaler()
        
        # Initialize
        await scaler.initialize()
        
        # Make predictions
        predictions = await scaler.predict_metrics(horizon_minutes=5)
        print(f"Predictions: {predictions}")
        
        # Make scaling decision
        decision = await scaler.make_scaling_decision("myapp-canary")
        if decision:
            print(f"Scaling Decision: {decision.direction.value} to {decision.target_value} replicas")
            print(f"Confidence: {decision.confidence:.2f}")
            print(f"Reasoning: {decision.reasoning}")
            
            # Execute scaling
            event = await scaler.execute_scaling_decision(decision, "myapp-canary")
            print(f"Scaling Event: {event.id} - Success: {event.success}")
        else:
            print("No scaling needed")
        
        # Get recommendations
        recommendations = scaler.get_scaling_recommendations("myapp-canary")
        print(f"Recommendations: {recommendations['recommendations']}")
    
    # Run test
    asyncio.run(test_predictive_scaler())

from kubernetes.client.rest import ApiException

# Monitoring
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
import requests
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    network_io: float
    disk_io: float
    request_rate: float
    response_time: float
    error_rate: float
    active_connections: int
    queue_length: int
    cost_per_hour: float

@dataclass
class ScalingPrediction:
    """Scaling prediction result"""
    timestamp: datetime
    predicted_load: float
    recommended_replicas: int
    recommended_cpu: float
    recommended_memory: float
    confidence_interval: Tuple[float, float]
    cost_impact: float
    scaling_reason: str
    urgency: str  # 'low', 'medium', 'high', 'critical'

@dataclass
class CloudResource:
    """Cloud resource configuration"""
    provider: str  # 'aws', 'gcp', 'azure'
    region: str
    instance_type: str
    cpu_cores: int
    memory_gb: float
    cost_per_hour: float
    availability: bool

class DeepARModel(nn.Module):
    """Deep AutoRegressive model for time series forecasting"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super(DeepARModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out[:, -1, :])
        return output

class PredictiveScaler:
    """AI-Powered Predictive Scaling Engine"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.prometheus_client = self._init_prometheus()
        self.redis_client = self._init_redis()
        self.k8s_client = self._init_kubernetes()
        
        # Cloud clients
        self.aws_client = self._init_aws()
        self.gcp_client = self._init_gcp()
        self.azure_client = self._init_azure()
        
        # ML Models
        self.prophet_models = {}
        self.deepar_model = DeepARModel(input_size=6)  # 6 features
        self.scaler = StandardScaler()
        self.cost_optimizer = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Metrics storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=2000))
        self.predictions_cache = {}
        
        # Cloud resources inventory
        self.cloud_resources = self._load_cloud_resources()
        
        # Prometheus metrics
        self.scaling_decisions = Counter(
            'scaling_decisions_total',
            'Total scaling decisions made',
            ['service', 'direction', 'reason']
        )
        self.prediction_accuracy = Gauge(
            'scaling_prediction_accuracy',
            'Accuracy of scaling predictions',
            ['service', 'model']
        )
        self.cost_savings = Counter(
            'cost_savings_total',
            'Total cost savings from predictive scaling',
            ['service']
        )
        
        logger.info("Predictive Scaler initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            'prometheus_url': os.getenv('PROMETHEUS_URL', 'http://prometheus-service:9090'),
            'redis_url': os.getenv('REDIS_URL', 'redis://redis-service:6379'),
            'prediction_horizon': int(os.getenv('PREDICTION_HORIZON', '3600')),  # 1 hour
            'scaling_threshold': float(os.getenv('SCALING_THRESHOLD', '0.8')),
            'min_replicas': int(os.getenv('MIN_REPLICAS', '2')),
            'max_replicas': int(os.getenv('MAX_REPLICAS', '100')),
            'cost_budget_hourly': float(os.getenv('COST_BUDGET_HOURLY', '1000.0')),
            'scaling_cooldown': int(os.getenv('SCALING_COOLDOWN', '300')),  # 5 minutes
            'enable_vertical_scaling': os.getenv('ENABLE_VERTICAL_SCALING', 'true').lower() == 'true',
            'enable_multi_cloud': os.getenv('ENABLE_MULTI_CLOUD', 'true').lower() == 'true',
            'aws_region': os.getenv('AWS_REGION', 'us-west-2'),
            'gcp_project': os.getenv('GCP_PROJECT', ''),
            'azure_subscription': os.getenv('AZURE_SUBSCRIPTION_ID', '')
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                import yaml
                file_config = yaml.safe_load(f)
                default_config.update(file_config)
        
        return default_config
    
    def _init_prometheus(self) -> requests.Session:
        """Initialize Prometheus client"""
        session = requests.Session()
        session.headers.update({'Content-Type': 'application/json'})
        return session
    
    def _init_redis(self) -> redis.Redis:
        """Initialize Redis client"""
        return redis.from_url(self.config['redis_url'])
    
    def _init_kubernetes(self) -> client.ApiClient:
        """Initialize Kubernetes client"""
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        return client.ApiClient()
    
    def _init_aws(self) -> boto3.Session:
        """Initialize AWS client"""
        try:
            return boto3.Session(region_name=self.config['aws_region'])
        except Exception as e:
            logger.warning(f"Failed to initialize AWS client: {e}")
            return None
    
    def _init_gcp(self):
        """Initialize GCP client"""
        try:
            if self.config['gcp_project']:
                return compute_v1.InstancesClient()
        except Exception as e:
            logger.warning(f"Failed to initialize GCP client: {e}")
        return None
    
    def _init_azure(self):
        """Initialize Azure client"""
        try:
            if self.config['azure_subscription']:
                credential = DefaultAzureCredential()
                return ComputeManagementClient(credential, self.config['azure_subscription'])
        except Exception as e:
            logger.warning(f"Failed to initialize Azure client: {e}")
        return None
    
    def _load_cloud_resources(self) -> List[CloudResource]:
        """Load available cloud resources"""
        resources = []
        
        # AWS instances
        aws_instances = [
            {'type': 't3.micro', 'cpu': 2, 'memory': 1, 'cost': 0.0104},
            {'type': 't3.small', 'cpu': 2, 'memory': 2, 'cost': 0.0208},
            {'type': 't3.medium', 'cpu': 2, 'memory': 4, 'cost': 0.0416},
            {'type': 't3.large', 'cpu': 2, 'memory': 8, 'cost': 0.0832},
            {'type': 'c5.large', 'cpu': 2, 'memory': 4, 'cost': 0.085},
            {'type': 'c5.xlarge', 'cpu': 4, 'memory': 8, 'cost': 0.17},
            {'type': 'm5.large', 'cpu': 2, 'memory': 8, 'cost': 0.096},
            {'type': 'm5.xlarge', 'cpu': 4, 'memory': 16, 'cost': 0.192}
        ]
        
        for instance in aws_instances:
            resources.append(CloudResource(
                provider='aws',
                region=self.config['aws_region'],
                instance_type=instance['type'],
                cpu_cores=instance['cpu'],
                memory_gb=instance['memory'],
                cost_per_hour=instance['cost'],
                availability=True
            ))
        
        # GCP instances
        gcp_instances = [
            {'type': 'e2-micro', 'cpu': 2, 'memory': 1, 'cost': 0.008},
            {'type': 'e2-small', 'cpu': 2, 'memory': 2, 'cost': 0.016},
            {'type': 'e2-medium', 'cpu': 2, 'memory': 4, 'cost': 0.032},
            {'type': 'n1-standard-1', 'cpu': 1, 'memory': 3.75, 'cost': 0.0475},
            {'type': 'n1-standard-2', 'cpu': 2, 'memory': 7.5, 'cost': 0.095},
            {'type': 'n1-standard-4', 'cpu': 4, 'memory': 15, 'cost': 0.19}
        ]
        
        for instance in gcp_instances:
            resources.append(CloudResource(
                provider='gcp',
                region='us-central1',
                instance_type=instance['type'],
                cpu_cores=instance['cpu'],
                memory_gb=instance['memory'],
                cost_per_hour=instance['cost'],
                availability=True
            ))
        
        return resources
    
    async def collect_metrics(self, service_name: str) -> ResourceMetrics:
        """Collect comprehensive resource metrics"""
        try:
            queries = {
                'cpu_usage': f'avg(rate(container_cpu_usage_seconds_total{{pod=~"{service_name}.*"}}[5m])) * 100',
                'memory_usage': f'avg(container_memory_usage_bytes{{pod=~"{service_name}.*"}}) / 1024 / 1024 / 1024',
                'network_io': f'sum(rate(container_network_receive_bytes_total{{pod=~"{service_name}.*"}}[5m])) + sum(rate(container_network_transmit_bytes_total{{pod=~"{service_name}.*"}}[5m]))',
                'disk_io': f'sum(rate(container_fs_reads_bytes_total{{pod=~"{service_name}.*"}}[5m])) + sum(rate(container_fs_writes_bytes_total{{pod=~"{service_name}.*"}}[5m]))',
                'request_rate': f'sum(rate(http_requests_total{{service="{service_name}"}}[5m]))',
                'response_time': f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{service="{service_name}"}}[5m])) by (le)) * 1000',
                'error_rate': f'sum(rate(http_requests_total{{service="{service_name}",status=~"5.."}}[5m])) / sum(rate(http_requests_total{{service="{service_name}"}}[5m]))',
                'active_connections': f'sum(http_active_connections{{service="{service_name}"}})',
                'queue_length': f'sum(queue_length{{service="{service_name}"}})',
            }
            
            metrics_data = {}
            for metric_name, query in queries.items():
                try:
                    response = self.prometheus_client.get(
                        f"{self.config['prometheus_url']}/api/v1/query",
                        params={'query': query}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data['data']['result']:
                            value = float(data['data']['result'][0]['value'][1])
                            metrics_data[metric_name] = value
                        else:
                            metrics_data[metric_name] = 0.0
                    else:
                        metrics_data[metric_name] = 0.0
                        
                except Exception as e:
                    logger.error(f"Error fetching {metric_name}: {e}")
                    metrics_data[metric_name] = 0.0
            
            # Calculate current cost
            current_cost = await self._calculate_current_cost(service_name)
            
            resource_metrics = ResourceMetrics(
                timestamp=datetime.now(),
                cpu_usage=metrics_data.get('cpu_usage', 0.0),
                memory_usage=metrics_data.get('memory_usage', 0.0),
                network_io=metrics_data.get('network_io', 0.0),
                disk_io=metrics_data.get('disk_io', 0.0),
                request_rate=metrics_data.get('request_rate', 0.0),
                response_time=metrics_data.get('response_time', 0.0),
                error_rate=metrics_data.get('error_rate', 0.0),
                active_connections=int(metrics_data.get('active_connections', 0)),
                queue_length=int(metrics_data.get('queue_length', 0)),
                cost_per_hour=current_cost
            )
            
            # Store in history
            self.metrics_history[service_name].append(resource_metrics)
            
            return resource_metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {service_name}: {e}")
            raise
    
    async def _calculate_current_cost(self, service_name: str) -> float:
        """Calculate current hourly cost for service"""
        try:
            # Get current pod count and resource allocation
            v1 = client.CoreV1Api()
            pods = v1.list_namespaced_pod(
                namespace='ai-news-dashboard',
                label_selector=f'app={service_name}'
            )
            
            total_cost = 0.0
            for pod in pods.items:
                if pod.spec.containers:
                    container = pod.spec.containers[0]
                    if container.resources and container.resources.requests:
                        cpu_request = container.resources.requests.get('cpu', '100m')
                        memory_request = container.resources.requests.get('memory', '128Mi')
                        
                        # Convert to standard units
                        cpu_cores = self._parse_cpu(cpu_request)
                        memory_gb = self._parse_memory(memory_request)
                        
                        # Find matching cloud resource
                        best_match = self._find_best_resource_match(cpu_cores, memory_gb)
                        if best_match:
                            total_cost += best_match.cost_per_hour
            
            return total_cost
            
        except Exception as e:
            logger.error(f"Error calculating current cost: {e}")
            return 0.0
    
    def _parse_cpu(self, cpu_str: str) -> float:
        """Parse CPU string to cores"""
        if cpu_str.endswith('m'):
            return float(cpu_str[:-1]) / 1000
        return float(cpu_str)
    
    def _parse_memory(self, memory_str: str) -> float:
        """Parse memory string to GB"""
        if memory_str.endswith('Mi'):
            return float(memory_str[:-2]) / 1024
        elif memory_str.endswith('Gi'):
            return float(memory_str[:-2])
        elif memory_str.endswith('Ki'):
            return float(memory_str[:-2]) / 1024 / 1024
        return float(memory_str) / 1024 / 1024 / 1024
    
    def _find_best_resource_match(self, cpu_cores: float, memory_gb: float) -> Optional[CloudResource]:
        """Find best matching cloud resource"""
        best_match = None
        best_score = float('inf')
        
        for resource in self.cloud_resources:
            if resource.cpu_cores >= cpu_cores and resource.memory_gb >= memory_gb:
                # Score based on resource efficiency and cost
                cpu_efficiency = cpu_cores / resource.cpu_cores
                memory_efficiency = memory_gb / resource.memory_gb
                efficiency_score = (cpu_efficiency + memory_efficiency) / 2
                cost_score = resource.cost_per_hour
                
                total_score = cost_score / efficiency_score
                
                if total_score < best_score:
                    best_score = total_score
                    best_match = resource
        
        return best_match
    
    async def predict_scaling_needs(self, service_name: str) -> ScalingPrediction:
        """Predict future scaling needs using ML models"""
        try:
            # Get historical metrics
            if service_name not in self.metrics_history or len(self.metrics_history[service_name]) < 50:
                logger.warning(f"Insufficient data for {service_name}, using default prediction")
                return self._default_prediction(service_name)
            
            metrics_list = list(self.metrics_history[service_name])
            
            # Prepare data for Prophet
            df = pd.DataFrame([
                {
                    'ds': m.timestamp,
                    'y': m.request_rate,
                    'cpu': m.cpu_usage,
                    'memory': m.memory_usage,
                    'response_time': m.response_time,
                    'error_rate': m.error_rate
                }
                for m in metrics_list
            ])
            
            # Train Prophet model if not exists or needs update
            if service_name not in self.prophet_models:
                self.prophet_models[service_name] = Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    changepoint_prior_scale=0.05
                )
                
                # Add additional regressors
                self.prophet_models[service_name].add_regressor('cpu')
                self.prophet_models[service_name].add_regressor('memory')
                self.prophet_models[service_name].add_regressor('response_time')
                self.prophet_models[service_name].add_regressor('error_rate')
                
                self.prophet_models[service_name].fit(df)
            
            # Make future predictions
            future_periods = self.config['prediction_horizon'] // 60  # Convert to minutes
            future = self.prophet_models[service_name].make_future_dataframe(
                periods=future_periods,
                freq='T'  # Minute frequency
            )
            
            # Add regressor values for future (use last known values)
            last_metrics = metrics_list[-1]
            future['cpu'] = last_metrics.cpu_usage
            future['memory'] = last_metrics.memory_usage
            future['response_time'] = last_metrics.response_time
            future['error_rate'] = last_metrics.error_rate
            
            forecast = self.prophet_models[service_name].predict(future)
            
            # Get prediction for target time
            target_time = datetime.now() + timedelta(seconds=self.config['prediction_horizon'])
            future_prediction = forecast.iloc[-1]
            
            predicted_load = max(0, future_prediction['yhat'])
            confidence_interval = (future_prediction['yhat_lower'], future_prediction['yhat_upper'])
            
            # Calculate recommended resources
            current_replicas = await self._get_current_replicas(service_name)
            recommended_replicas, recommended_cpu, recommended_memory = await self._calculate_resource_requirements(
                service_name, predicted_load, last_metrics
            )
            
            # Determine scaling urgency
            load_increase = predicted_load / max(1, last_metrics.request_rate)
            if load_increase > 2.0:
                urgency = 'critical'
                scaling_reason = 'Predicted traffic spike > 200%'
            elif load_increase > 1.5:
                urgency = 'high'
                scaling_reason = 'Predicted traffic increase > 150%'
            elif load_increase > 1.2:
                urgency = 'medium'
                scaling_reason = 'Predicted traffic increase > 120%'
            else:
                urgency = 'low'
                scaling_reason = 'Normal traffic pattern'
            
            # Calculate cost impact
            current_cost = last_metrics.cost_per_hour * current_replicas
            predicted_cost = await self._estimate_cost(recommended_replicas, recommended_cpu, recommended_memory)
            cost_impact = predicted_cost - current_cost
            
            prediction = ScalingPrediction(
                timestamp=target_time,
                predicted_load=predicted_load,
                recommended_replicas=recommended_replicas,
                recommended_cpu=recommended_cpu,
                recommended_memory=recommended_memory,
                confidence_interval=confidence_interval,
                cost_impact=cost_impact,
                scaling_reason=scaling_reason,
                urgency=urgency
            )
            
            # Cache prediction
            self.predictions_cache[service_name] = prediction
            
            # Store in Redis
            await self._store_prediction(service_name, prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting scaling needs for {service_name}: {e}")
            return self._default_prediction(service_name)
    
    def _default_prediction(self, service_name: str) -> ScalingPrediction:
        """Return default prediction when insufficient data"""
        return ScalingPrediction(
            timestamp=datetime.now() + timedelta(seconds=self.config['prediction_horizon']),
            predicted_load=0.0,
            recommended_replicas=self.config['min_replicas'],
            recommended_cpu=0.5,
            recommended_memory=1.0,
            confidence_interval=(0.0, 0.0),
            cost_impact=0.0,
            scaling_reason='Insufficient historical data',
            urgency='low'
        )
    
    async def _get_current_replicas(self, service_name: str) -> int:
        """Get current number of replicas"""
        try:
            apps_v1 = client.AppsV1Api()
            deployment = apps_v1.read_namespaced_deployment(
                name=service_name,
                namespace='ai-news-dashboard'
            )
            return deployment.status.replicas or 1
        except Exception as e:
            logger.error(f"Error getting current replicas: {e}")
            return 1
    
    async def _calculate_resource_requirements(self, service_name: str, predicted_load: float, current_metrics: ResourceMetrics) -> Tuple[int, float, float]:
        """Calculate required resources based on predicted load"""
        try:
            current_replicas = await self._get_current_replicas(service_name)
            current_load = current_metrics.request_rate
            
            if current_load > 0:
                load_ratio = predicted_load / current_load
            else:
                load_ratio = 1.0
            
            # Calculate required replicas with safety margin
            safety_margin = 1.2  # 20% safety margin
            required_replicas = max(
                self.config['min_replicas'],
                min(
                    self.config['max_replicas'],
                    int(current_replicas * load_ratio * safety_margin)
                )
            )
            
            # Calculate CPU and memory requirements
            cpu_utilization_target = 0.7  # Target 70% CPU utilization
            memory_utilization_target = 0.8  # Target 80% memory utilization
            
            if current_metrics.cpu_usage > 0:
                cpu_ratio = current_metrics.cpu_usage / 100.0
                recommended_cpu = max(0.1, (cpu_ratio / cpu_utilization_target) * load_ratio)
            else:
                recommended_cpu = 0.5
            
            if current_metrics.memory_usage > 0:
                memory_ratio = current_metrics.memory_usage / 8.0  # Assume 8GB base
                recommended_memory = max(0.5, (memory_ratio / memory_utilization_target) * load_ratio)
            else:
                recommended_memory = 1.0
            
            return required_replicas, recommended_cpu, recommended_memory
            
        except Exception as e:
            logger.error(f"Error calculating resource requirements: {e}")
            return self.config['min_replicas'], 0.5, 1.0
    
    async def _estimate_cost(self, replicas: int, cpu: float, memory: float) -> float:
        """Estimate hourly cost for given resources"""
        try:
            best_resource = self._find_best_resource_match(cpu, memory)
            if best_resource:
                return replicas * best_resource.cost_per_hour
            return replicas * 0.1  # Default cost estimate
        except Exception as e:
            logger.error(f"Error estimating cost: {e}")
            return 0.0
    
    async def _store_prediction(self, service_name: str, prediction: ScalingPrediction):
        """Store prediction in Redis"""
        try:
            key = f"scaling_prediction:{service_name}:{int(time.time())}"
            value = json.dumps(asdict(prediction), default=str)
            self.redis_client.setex(key, 3600, value)
            
            # Store latest prediction
            latest_key = f"scaling_prediction:latest:{service_name}"
            self.redis_client.setex(latest_key, 3600, value)
            
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
    
    async def execute_scaling(self, service_name: str, prediction: ScalingPrediction) -> bool:
        """Execute scaling decision"""
        try:
            current_replicas = await self._get_current_replicas(service_name)
            
            # Check if scaling is needed
            if abs(prediction.recommended_replicas - current_replicas) <= 1:
                logger.info(f"No scaling needed for {service_name}")
                return True
            
            # Check cost constraints
            if prediction.cost_impact > self.config['cost_budget_hourly'] * 0.1:  # 10% of budget
                logger.warning(f"Scaling {service_name} would exceed cost budget")
                return False
            
            # Execute horizontal scaling
            success = await self._scale_deployment(
                service_name,
                prediction.recommended_replicas
            )
            
            if success and self.config['enable_vertical_scaling']:
                # Execute vertical scaling
                await self._update_resource_limits(
                    service_name,
                    prediction.recommended_cpu,
                    prediction.recommended_memory
                )
            
            # Update metrics
            direction = 'up' if prediction.recommended_replicas > current_replicas else 'down'
            self.scaling_decisions.labels(
                service=service_name,
                direction=direction,
                reason=prediction.scaling_reason
            ).inc()
            
            if prediction.cost_impact < 0:
                self.cost_savings.labels(service=service_name).inc(abs(prediction.cost_impact))
            
            logger.info(f"Successfully scaled {service_name} from {current_replicas} to {prediction.recommended_replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Error executing scaling for {service_name}: {e}")
            return False
    
    async def _scale_deployment(self, service_name: str, target_replicas: int) -> bool:
        """Scale Kubernetes deployment"""
        try:
            apps_v1 = client.AppsV1Api()
            
            # Update deployment replicas
            body = {'spec': {'replicas': target_replicas}}
            apps_v1.patch_namespaced_deployment(
                name=service_name,
                namespace='ai-news-dashboard',
                body=body
            )
            
            logger.info(f"Scaled {service_name} to {target_replicas} replicas")
            return True
            
        except ApiException as e:
            logger.error(f"Kubernetes API error scaling {service_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error scaling {service_name}: {e}")
            return False
    
    async def _update_resource_limits(self, service_name: str, cpu: float, memory: float) -> bool:
        """Update resource limits for deployment"""
        try:
            apps_v1 = client.AppsV1Api()
            
            # Update resource limits
            body = {
                'spec': {
                    'template': {
                        'spec': {
                            'containers': [{
                                'name': service_name,
                                'resources': {
                                    'requests': {
                                        'cpu': f'{cpu}',
                                        'memory': f'{memory}Gi'
                                    },
                                    'limits': {
                                        'cpu': f'{cpu * 1.5}',
                                        'memory': f'{memory * 1.2}Gi'
                                    }
                                }
                            }]
                        }
                    }
                }
            }
            
            apps_v1.patch_namespaced_deployment(
                name=service_name,
                namespace='ai-news-dashboard',
                body=body
            )
            
            logger.info(f"Updated resource limits for {service_name}: CPU={cpu}, Memory={memory}Gi")
            return True
            
        except Exception as e:
            logger.error(f"Error updating resource limits for {service_name}: {e}")
            return False

async def main():
    """Main execution function"""
    scaler = PredictiveScaler()
    
    # Start Prometheus metrics server
    prometheus_client.start_http_server(8091)
    
    # Continuous scaling loop
    while True:
        try:
            services = ['api-service', 'nlp-service', 'cv-service', 'frontend']
            
            for service in services:
                try:
                    # Collect metrics
                    metrics = await scaler.collect_metrics(service)
                    
                    # Predict scaling needs
                    prediction = await scaler.predict_scaling_needs(service)
                    
                    # Execute scaling if needed
                    if prediction.urgency in ['high', 'critical']:
                        success = await scaler.execute_scaling(service, prediction)
                        if success:
                            logger.info(f"Executed scaling for {service}: {prediction.scaling_reason}")
                        else:
                            logger.error(f"Failed to execute scaling for {service}")
                    
                except Exception as e:
                    logger.error(f"Error processing {service}: {e}")
            
            # Wait before next cycle
            await asyncio.sleep(120)  # Check every 2 minutes
            
        except KeyboardInterrupt:
            logger.info("Shutting down Predictive Scaler")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())