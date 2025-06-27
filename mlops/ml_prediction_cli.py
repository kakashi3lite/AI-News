#!/usr/bin/env python3
"""
ML Prediction Service CLI
Command-line interface for running ML prediction tasks from the scheduler

Usage:
    python ml_prediction_cli.py --task=trend_forecasting --horizon=24h
    python ml_prediction_cli.py --task=popularity_prediction
    python ml_prediction_cli.py --task=sentiment_trends
    python ml_prediction_cli.py --task=topic_emergence
"""

import argparse
import asyncio
import logging
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from ml_prediction_service import (
        MLPredictionService,
        PredictionRequest,
        PredictionType
    )
except ImportError as e:
    logger.error(f"Failed to import ML prediction service: {e}")
    sys.exit(1)

class MLPredictionCLI:
    """Command-line interface for ML prediction service"""
    
    def __init__(self):
        self.service = None
        self.results = {}
    
    async def initialize_service(self, config: Optional[Dict] = None):
        """Initialize the ML prediction service"""
        try:
            logger.info("🚀 Initializing ML Prediction Service...")
            self.service = MLPredictionService(config)
            await self.service.initialize()
            logger.info("✅ ML Prediction Service initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML Prediction Service: {e}")
            raise
    
    async def run_trend_forecasting(self, horizon: str = "24h", model_version: str = "v1.0.0") -> Dict[str, Any]:
        """Run trend forecasting task"""
        logger.info(f"📈 Running trend forecasting for {horizon} horizon...")
        
        try:
            # Parse horizon
            if horizon.endswith('h'):
                hours = int(horizon[:-1])
            elif horizon.endswith('d'):
                hours = int(horizon[:-1]) * 24
            else:
                hours = 24  # Default to 24 hours
            
            # Create prediction request
            request = PredictionRequest(
                prediction_type="trend",
                input_data={
                    'cli_execution': True,
                    'scheduled_task': True,
                    'horizon_requested': horizon
                },
                time_horizon=f"{hours}h",
                model_version=model_version
            )
            
            # Run prediction
            result = await self.service.predict(request)
            
            # Process and log results
            logger.info(f"✅ Trend forecasting completed successfully")
            logger.info(f"  Prediction: {result.predicted_value}")
            logger.info(f"  Confidence: {result.confidence_score:.2f}")
            logger.info(f"  Model: {result.model_used}")
            
            return {
                'task': 'trend_forecasting',
                'status': 'success',
                'predicted_value': result.predicted_value,
                'confidence_score': result.confidence_score,
                'model_used': result.model_used,
                'model_version': result.model_version
            }
            
        except Exception as e:
            logger.error(f"❌ Trend forecasting failed: {e}")
            return {
                'task': 'trend_forecasting',
                'status': 'error',
                'error': str(e)
            }
    
    async def run_popularity_prediction(self) -> Dict[str, Any]:
        """Run popularity prediction task"""
        logger.info("🔥 Running popularity prediction...")
        
        try:
            # Create prediction request
            request = PredictionRequest(
                prediction_type="popularity",
                input_data={
                    'cli_execution': True,
                    'scheduled_task': True
                }
            )
            
            # Run prediction
            result = await self.service.predict(request)
            
            # Process results
            logger.info(f"✅ Popularity prediction completed successfully")
            logger.info(f"  Prediction: {result.predicted_value}")
            logger.info(f"  Confidence: {result.confidence_score:.2f}")
            logger.info(f"  Model: {result.model_used}")
            
            return {
                'task': 'popularity_prediction',
                'status': 'success',
                'predicted_value': result.predicted_value,
                'confidence_score': result.confidence_score,
                'model_used': result.model_used,
                'processing_time': result.processing_time,
                'model_version': result.model_version
            }
            
        except Exception as e:
            logger.error(f"❌ Popularity prediction failed: {e}")
            return {
                'task': 'popularity_prediction',
                'status': 'error',
                'error': str(e)
            }
    
    async def run_sentiment_trends(self) -> Dict[str, Any]:
        """Run sentiment trends analysis"""
        logger.info("😊 Running sentiment trends analysis...")
        
        try:
            # Create prediction request
            request = PredictionRequest(
                prediction_type="sentiment",
                input_data={
                    'cli_execution': True,
                    'scheduled_task': True
                }
            )
            
            # Run prediction
            result = await self.service.predict(request)
            
            # Process results
            logger.info(f"✅ Sentiment trend analysis completed successfully")
            logger.info(f"  Prediction: {result.predicted_value}")
            logger.info(f"  Confidence: {result.confidence_score:.2f}")
            logger.info(f"  Model: {result.model_used}")
            
            return {
                'task': 'sentiment_trends',
                'status': 'success',
                'predicted_value': result.predicted_value,
                'confidence_score': result.confidence_score,
                'model_used': result.model_used,
                'model_version': result.model_version
            }
            
        except Exception as e:
            logger.error(f"❌ Sentiment trends analysis failed: {e}")
            return {
                'task': 'sentiment_trends',
                'status': 'error',
                'error': str(e)
            }
    
    async def run_topic_emergence(self) -> Dict[str, Any]:
        """Run topic emergence detection"""
        logger.info("🆕 Running topic emergence detection...")
        
        try:
            # Create prediction request
            request = PredictionRequest(
                prediction_type="topic_emergence",
                input_data={
                    'texts': [
                        "Breaking news in artificial intelligence and machine learning developments",
                        "Political developments affecting global markets and trade policies",
                        "Healthcare innovations and medical research breakthroughs",
                        "Environmental concerns and climate change initiatives",
                        "Technology sector growth and digital transformation trends"
                    ],
                    'cli_execution': True,
                    'scheduled_task': True
                }
            )
            
            # Run prediction
            result = await self.service.predict(request)
            
            # Process results
            logger.info(f"✅ Topic emergence detection completed successfully")
            logger.info(f"  Prediction: {result.predicted_value}")
            logger.info(f"  Confidence: {result.confidence_score:.2f}")
            logger.info(f"  Model: {result.model_used}")
            
            return {
                'task': 'topic_emergence',
                'status': 'success',
                'predicted_value': result.predicted_value,
                'confidence_score': result.confidence_score,
                'model_used': result.model_used,
                'model_version': result.model_version
            }
            
        except Exception as e:
            logger.error(f"❌ Topic emergence detection failed: {e}")
            return {
                'task': 'topic_emergence',
                'status': 'error',
                'error': str(e)
            }
    
    async def run_user_engagement_prediction(self) -> Dict[str, Any]:
        """Run user engagement prediction"""
        logger.info("👥 Running user engagement prediction...")
        
        try:
            # Create prediction request
            request = PredictionRequest(
                prediction_type="engagement",
                input_data={
                    'cli_execution': True,
                    'scheduled_task': True
                }
            )
            
            # Run prediction
            result = await self.service.predict(request)
            
            # Process results
            logger.info(f"✅ User engagement prediction completed successfully")
            logger.info(f"  Prediction: {result.predicted_value}")
            logger.info(f"  Confidence: {result.confidence_score:.2f}")
            logger.info(f"  Model: {result.model_used}")
            
            return {
                'task': 'user_engagement_prediction',
                'status': 'success',
                'predicted_value': result.predicted_value,
                'confidence_score': result.confidence_score,
                'model_used': result.model_used,
                'model_version': result.model_version
            }
            
        except Exception as e:
            logger.error(f"❌ User engagement prediction failed: {e}")
            return {
                'task': 'user_engagement_prediction',
                'status': 'error',
                'error': str(e)
            }
    
    async def run_task(self, task: str, **kwargs) -> Dict[str, Any]:
        """Run a specific ML prediction task"""
        task_map = {
            'trend_forecasting': self.run_trend_forecasting,
            'popularity_prediction': self.run_popularity_prediction,
            'sentiment_trends': self.run_sentiment_trends,
            'topic_emergence': self.run_topic_emergence,
            'user_engagement': self.run_user_engagement_prediction
        }
        
        if task not in task_map:
            raise ValueError(f"Unknown task: {task}. Available tasks: {list(task_map.keys())}")
        
        # Initialize service if not already done
        if not self.service:
            await self.initialize_service()
        
        # Run the task
        if task == 'trend_forecasting':
            return await task_map[task](
                horizon=kwargs.get('horizon', '24h'),
                model_version=kwargs.get('model_version', 'v1.0.0')
            )
        else:
            return await task_map[task]()
    
    def save_results(self, results: Dict[str, Any], output_file: Optional[str] = None):
        """Save results to file"""
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"📄 Results saved to {output_file}")
            except Exception as e:
                logger.error(f"❌ Failed to save results: {e}")
        
        # Always save to default location
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_file = f"ml_prediction_results_{timestamp}.json"
        
        try:
            with open(default_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"📄 Results also saved to {default_file}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save to default file: {e}")

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='ML Prediction Service CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ml_prediction_cli.py --task=trend_forecasting --horizon=24h
  python ml_prediction_cli.py --task=popularity_prediction
  python ml_prediction_cli.py --task=sentiment_trends
  python ml_prediction_cli.py --task=topic_emergence
  python ml_prediction_cli.py --task=user_engagement
        """
    )
    
    parser.add_argument(
        '--task',
        required=True,
        choices=['trend_forecasting', 'popularity_prediction', 'sentiment_trends', 'topic_emergence', 'user_engagement'],
        help='ML prediction task to run'
    )
    
    parser.add_argument(
        '--horizon',
        default='24h',
        help='Time horizon for trend forecasting (e.g., 24h, 7d)'
    )
    
    parser.add_argument(
        '--model-version',
        default='v1.0.0',
        help='Model version to use'
    )
    
    parser.add_argument(
        '--output',
        help='Output file for results (JSON format)'
    )
    
    parser.add_argument(
        '--config',
        help='Configuration file path (JSON format)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

async def main():
    """Main CLI function"""
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"📋 Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            sys.exit(1)
    
    # Initialize CLI
    cli = MLPredictionCLI()
    
    try:
        # Run the specified task
        logger.info(f"🎯 Starting ML prediction task: {args.task}")
        start_time = datetime.now()
        
        results = await cli.run_task(
            task=args.task,
            horizon=args.horizon,
            model_version=getattr(args, 'model_version', 'v1.0.0')
        )
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Add execution metadata
        results['execution_metadata'] = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_execution_time': total_time,
            'cli_version': '1.0.0',
            'arguments': vars(args)
        }
        
        # Save results
        cli.save_results(results, args.output)
        
        # Print summary
        logger.info(f"🎉 Task '{args.task}' completed successfully in {total_time:.2f}s")
        logger.info(f"📊 Status: {results.get('status', 'unknown')}")
        
        if results.get('status') == 'success':
            sys.exit(0)
        else:
            logger.error(f"❌ Task failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Task interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())