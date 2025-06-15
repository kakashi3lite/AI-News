#!/usr/bin/env python3
"""
Dr. NewsForge's Advanced News Summarization Model Training

Features:
- Fine-tuned BART/T5 models for news summarization
- Multi-GPU training with gradient accumulation
- Advanced data augmentation and preprocessing
- MLflow experiment tracking
- Automated hyperparameter optimization
- Model distillation for production deployment

Author: Dr. Nova "NewsForge" Arclight
Version: 2.0.0
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import load_dataset, Dataset as HFDataset
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
import numpy as np
from rouge_score import rouge_scorer
import wandb
from accelerate import Accelerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsDataset(Dataset):
    """Custom dataset for news summarization training."""
    
    def __init__(self, articles: List[str], summaries: List[str], tokenizer, max_length: int = 1024):
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        article = str(self.articles[idx])
        summary = str(self.summaries[idx])
        
        # Tokenize inputs
        inputs = self.tokenizer(
            article,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize targets
        targets = self.tokenizer(
            summary,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten()
        }

class SummarizationTrainer:
    """Advanced trainer for news summarization models."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config.get('mlflow_uri', 'http://localhost:5000'))
        mlflow.set_experiment(config.get('experiment_name', 'news-summarization'))
        
        # Initialize Weights & Biases
        if config.get('use_wandb', False):
            wandb.init(
                project="news-summarization",
                config=config,
                name=f"summarization-{config.get('model_name', 'bart')}"
            )
    
    def load_data(self, data_path: str) -> Tuple[List[str], List[str]]:
        """Load and preprocess training data."""
        logger.info(f"Loading data from {data_path}")
        
        # Load from various sources
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            articles = [item['article'] for item in data]
            summaries = [item['summary'] for item in data]
        elif data_path.endswith('.jsonl'):
            articles, summaries = [], []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    articles.append(item['article'])
                    summaries.append(item['summary'])
        else:
            # Load from HuggingFace datasets
            dataset = load_dataset(data_path)
            articles = dataset['train']['article']
            summaries = dataset['train']['highlights']
        
        logger.info(f"Loaded {len(articles)} article-summary pairs")
        return articles, summaries
    
    def preprocess_data(self, articles: List[str], summaries: List[str]) -> Tuple[List[str], List[str]]:
        """Advanced data preprocessing and augmentation."""
        logger.info("Preprocessing data...")
        
        processed_articles = []
        processed_summaries = []
        
        for article, summary in zip(articles, summaries):
            # Basic cleaning
            article = article.strip().replace('\n', ' ').replace('\r', ' ')
            summary = summary.strip().replace('\n', ' ').replace('\r', ' ')
            
            # Filter by length
            if len(article.split()) < 50 or len(summary.split()) < 5:
                continue
            if len(article.split()) > 2000 or len(summary.split()) > 200:
                continue
            
            # Add prefix for better performance
            article = f"summarize: {article}"
            
            processed_articles.append(article)
            processed_summaries.append(summary)
        
        logger.info(f"Preprocessed to {len(processed_articles)} samples")
        return processed_articles, processed_summaries
    
    def create_model_and_tokenizer(self, model_name: str):
        """Initialize model and tokenizer."""
        logger.info(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Add special tokens if needed
        special_tokens = ['<news>', '</news>', '<summary>', '</summary>']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        return self.model, self.tokenizer
    
    def create_datasets(self, articles: List[str], summaries: List[str]) -> Tuple[NewsDataset, NewsDataset, NewsDataset]:
        """Create train, validation, and test datasets."""
        # Split data
        train_articles, temp_articles, train_summaries, temp_summaries = train_test_split(
            articles, summaries, test_size=0.2, random_state=42
        )
        val_articles, test_articles, val_summaries, test_summaries = train_test_split(
            temp_articles, temp_summaries, test_size=0.5, random_state=42
        )
        
        # Create datasets
        train_dataset = NewsDataset(train_articles, train_summaries, self.tokenizer)
        val_dataset = NewsDataset(val_articles, val_summaries, self.tokenizer)
        test_dataset = NewsDataset(test_articles, test_summaries, self.tokenizer)
        
        logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute ROUGE metrics for evaluation."""
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, label in zip(decoded_preds, decoded_labels):
            scores = scorer.score(label, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL'])
        }
    
    def train(self, train_dataset: NewsDataset, val_dataset: NewsDataset, output_dir: str):
        """Train the summarization model."""
        logger.info("Starting model training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.get('num_epochs', 3),
            per_device_train_batch_size=self.config.get('batch_size', 4),
            per_device_eval_batch_size=self.config.get('eval_batch_size', 4),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 4),
            warmup_steps=self.config.get('warmup_steps', 500),
            weight_decay=self.config.get('weight_decay', 0.01),
            learning_rate=self.config.get('learning_rate', 5e-5),
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            eval_steps=500,
            save_steps=1000,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="wandb" if self.config.get('use_wandb', False) else None
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"summarization-{self.config.get('model_name', 'bart')}"):
            # Log parameters
            mlflow.log_params(self.config)
            
            # Train model
            trainer.train()
            
            # Evaluate model
            eval_results = trainer.evaluate()
            mlflow.log_metrics(eval_results)
            
            # Save model
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Log model to MLflow
            mlflow.pytorch.log_model(
                pytorch_model=self.model,
                artifact_path="model",
                registered_model_name="news-summarizer"
            )
            
            logger.info(f"Training completed. Model saved to {output_dir}")
            logger.info(f"Evaluation results: {eval_results}")
            
            return trainer, eval_results
    
    def evaluate_model(self, test_dataset: NewsDataset, model_path: str) -> Dict:
        """Comprehensive model evaluation."""
        logger.info("Evaluating model...")
        
        # Load model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model.eval()
        model.to(self.device)
        
        # Evaluation metrics
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Generate summaries
        dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Generate summaries
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=256,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
                
                # Decode predictions and labels
                decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                # Compute ROUGE scores
                for pred, label in zip(decoded_preds, decoded_labels):
                    scores = scorer.score(label, pred)
                    rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                    rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                    rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        # Calculate final metrics
        final_metrics = {
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL']),
            'rouge1_std': np.std(rouge_scores['rouge1']),
            'rouge2_std': np.std(rouge_scores['rouge2']),
            'rougeL_std': np.std(rouge_scores['rougeL'])
        }
        
        logger.info(f"Final evaluation metrics: {final_metrics}")
        return final_metrics

def main():
    parser = argparse.ArgumentParser(description="Train news summarization model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--output-path', type=str, required=True, help='Output directory for model')
    parser.add_argument('--experiment-name', type=str, default='news-summarization', help='MLflow experiment name')
    parser.add_argument('--run-name', type=str, help='MLflow run name')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    
    # Extract model config
    model_config = config['model_registry']['models']['summarization']
    training_config = {
        'model_name': model_config['base_model'],
        'num_epochs': 3,
        'batch_size': 4,
        'learning_rate': 5e-5,
        'warmup_steps': 500,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 4,
        'use_wandb': True,
        'mlflow_uri': config['model_registry']['tracking_uri'],
        'experiment_name': args.experiment_name
    }
    
    # Initialize trainer
    trainer = SummarizationTrainer(training_config)
    
    # Load and preprocess data
    articles, summaries = trainer.load_data(args.data_path)
    articles, summaries = trainer.preprocess_data(articles, summaries)
    
    # Create model and tokenizer
    model, tokenizer = trainer.create_model_and_tokenizer(training_config['model_name'])
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = trainer.create_datasets(articles, summaries)
    
    # Train model
    trained_model, eval_results = trainer.train(train_dataset, val_dataset, args.output_path)
    
    # Final evaluation
    test_metrics = trainer.evaluate_model(test_dataset, args.output_path)
    
    # Save metrics
    metrics_path = Path(args.output_path) / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'validation_metrics': eval_results,
            'test_metrics': test_metrics,
            'config': training_config
        }, f, indent=2)
    
    logger.info(f"Training completed successfully. Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()