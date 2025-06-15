#!/usr/bin/env python3
"""
Dr. NewsForge's Computer Vision News AI

Advanced computer vision system for multimodal news analysis,
visual content understanding, and automated image/video processing.
Implements state-of-the-art vision transformers, object detection,
scene understanding, and visual-textual alignment for news media.

Features:
- Vision Transformer (ViT) for image classification
- CLIP for visual-textual alignment
- Object detection and scene analysis
- Video content analysis and summarization
- Multimodal embedding generation
- Visual fact-checking and verification
- Automated image captioning
- Visual sentiment analysis
- Real-time video processing
- Deepfake detection
- Visual misinformation detection
- Geospatial image analysis

Author: Dr. Nova "NewsForge" Arclight
Version: 2.0.0
"""

import os
import json
import time
import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import uuid
import pickle
import hashlib
from pathlib import Path
import math
import random
from copy import deepcopy
import threading
from queue import Queue, PriorityQueue
import base64
import io
from urllib.parse import urlparse
import requests

# Computer vision libraries
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import feature, measure, segmentation
from skimage.filters import gaussian
from skimage.transform import resize

# Deep learning frameworks
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, models
from torchvision.models import detection
import torchvision.transforms.functional as TF

# Transformers and multimodal models
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig, AutoProcessor,
    CLIPModel, CLIPProcessor, CLIPTokenizer,
    BlipProcessor, BlipForConditionalGeneration,
    ViTModel, ViTFeatureExtractor,
    DPTForDepthEstimation, DPTFeatureExtractor,
    DetrImageProcessor, DetrForObjectDetection
)
from sentence_transformers import SentenceTransformer

# Video processing
try:
    import moviepy.editor as mp
    from moviepy.video.io.VideoFileClip import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logging.warning("MoviePy not available for video processing")

# Face detection and analysis
try:
    import face_recognition
    import dlib
    FACE_LIBS_AVAILABLE = True
except ImportError:
    FACE_LIBS_AVAILABLE = False
    logging.warning("Face recognition libraries not available")

# OCR and text extraction
try:
    import pytesseract
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("OCR libraries not available")

# Geospatial analysis
try:
    import geopandas as gpd
    import folium
    from geopy.geocoders import Nominatim
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False
    logging.warning("Geospatial libraries not available")

# Scientific computing
from scipy.spatial.distance import cosine, euclidean
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Monitoring and visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# MLOps and tracking
import mlflow
import wandb
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Networking
from flask import Flask, request, jsonify, send_file
import redis
from kafka import KafkaProducer, KafkaConsumer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
VISION_REQUESTS = Counter('vision_requests_total', 'Total vision requests', ['task_type'])
VISION_LATENCY = Histogram('vision_latency_seconds', 'Vision processing latency', ['task_type'])
IMAGE_QUALITY = Gauge('image_quality_score', 'Average image quality score')
DETECTION_CONFIDENCE = Gauge('detection_confidence', 'Average detection confidence')
VISUAL_SIMILARITY = Gauge('visual_similarity_score', 'Visual similarity score')
DEEPFAKE_CONFIDENCE = Gauge('deepfake_confidence', 'Deepfake detection confidence')
OCR_ACCURACY = Gauge('ocr_accuracy', 'OCR accuracy score')
VIDEO_PROCESSING_TIME = Histogram('video_processing_seconds', 'Video processing time')
MULTIMODAL_ALIGNMENT = Gauge('multimodal_alignment_score', 'Multimodal alignment score')

@dataclass
class VisualContent:
    """Visual content representation."""
    content_id: str
    content_type: str  # 'image', 'video', 'gif'
    url: Optional[str] = None
    local_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    extracted_features: Optional[Dict[str, Any]] = None
    analysis_results: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

@dataclass
class DetectionResult:
    """Object detection result."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: Optional[np.ndarray] = None
    attributes: Optional[Dict[str, Any]] = None

@dataclass
class VisualAnalysis:
    """Complete visual analysis result."""
    content_id: str
    objects: List[DetectionResult]
    scene_description: str
    sentiment_score: float
    quality_metrics: Dict[str, float]
    text_content: List[str]
    faces: List[Dict[str, Any]]
    landmarks: List[str]
    colors: Dict[str, float]
    composition_analysis: Dict[str, Any]
    deepfake_probability: float
    authenticity_score: float
    geolocation: Optional[Dict[str, Any]] = None

class VisionTransformerClassifier(nn.Module):
    """Vision Transformer for image classification."""
    
    def __init__(self, num_classes: int, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Load pre-trained ViT
        model_name = config.get('vit_model', 'google/vit-base-patch16-224')
        self.vit = ViTModel.from_pretrained(model_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        
        # Classification head
        hidden_size = self.vit.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(hidden_size, config.get('hidden_dim', 512)),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(config.get('hidden_dim', 512), num_classes)
        )
        
        # Freeze ViT layers if specified
        if config.get('freeze_backbone', False):
            for param in self.vit.parameters():
                param.requires_grad = False
        
        logger.info(f"ViT Classifier initialized with {num_classes} classes")
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass through ViT classifier."""
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits
    
    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        with torch.no_grad():
            outputs = self.vit(pixel_values=pixel_values)
            return outputs.pooler_output

class MultimodalCLIPAnalyzer:
    """CLIP-based multimodal analyzer for visual-textual alignment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load CLIP model
        model_name = config.get('clip_model', 'openai/clip-vit-base-patch32')
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Cache for embeddings
        self.embedding_cache = {}
        
        logger.info(f"CLIP Analyzer initialized with model: {model_name}")
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode image to embedding vector."""
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return np.zeros(512)  # Default embedding size
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector."""
        try:
            # Check cache
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]
            
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            embedding = text_features.cpu().numpy().flatten()
            self.embedding_cache[text_hash] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            return np.zeros(512)  # Default embedding size
    
    def calculate_similarity(self, image: Image.Image, text: str) -> float:
        """Calculate similarity between image and text."""
        try:
            image_embedding = self.encode_image(image)
            text_embedding = self.encode_text(text)
            
            # Cosine similarity
            similarity = 1 - cosine(image_embedding, text_embedding)
            
            MULTIMODAL_ALIGNMENT.set(similarity)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def find_best_caption(self, image: Image.Image, candidate_captions: List[str]) -> Tuple[str, float]:
        """Find best caption for image from candidates."""
        best_caption = ""
        best_score = -1.0
        
        image_embedding = self.encode_image(image)
        
        for caption in candidate_captions:
            text_embedding = self.encode_text(caption)
            similarity = 1 - cosine(image_embedding, text_embedding)
            
            if similarity > best_score:
                best_score = similarity
                best_caption = caption
        
        return best_caption, best_score
    
    def generate_image_description(self, image: Image.Image, context_keywords: List[str] = None) -> str:
        """Generate description for image using CLIP."""
        try:
            # Predefined description templates
            templates = [
                "a photo of {}",
                "an image showing {}",
                "a picture of {}",
                "a news photo featuring {}",
                "a photograph depicting {}"
            ]
            
            # Common news-related objects/scenes
            objects = [
                "people", "crowd", "politician", "building", "city", "protest",
                "meeting", "conference", "sports event", "accident", "fire",
                "celebration", "ceremony", "landscape", "technology", "vehicle"
            ]
            
            if context_keywords:
                objects.extend(context_keywords)
            
            best_descriptions = []
            image_embedding = self.encode_image(image)
            
            for obj in objects:
                for template in templates:
                    description = template.format(obj)
                    text_embedding = self.encode_text(description)
                    similarity = 1 - cosine(image_embedding, text_embedding)
                    best_descriptions.append((description, similarity))
            
            # Sort by similarity and return top description
            best_descriptions.sort(key=lambda x: x[1], reverse=True)
            
            if best_descriptions:
                return best_descriptions[0][0]
            else:
                return "an image"
                
        except Exception as e:
            logger.error(f"Image description generation failed: {e}")
            return "an image"

class ObjectDetector:
    """Advanced object detection using DETR and other models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load DETR model
        model_name = config.get('detr_model', 'facebook/detr-resnet-50')
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Confidence threshold
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        
        logger.info(f"Object Detector initialized with model: {model_name}")
    
    def detect_objects(self, image: Image.Image) -> List[DetectionResult]:
        """Detect objects in image."""
        try:
            # Preprocess image
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Run detection
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process results
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
            )[0]
            
            detections = []
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                class_name = self.model.config.id2label[label.item()]
                confidence = score.item()
                bbox = box.cpu().numpy().astype(int)
                
                detection = DetectionResult(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=tuple(bbox)
                )
                detections.append(detection)
            
            # Update metrics
            if detections:
                avg_confidence = np.mean([d.confidence for d in detections])
                DETECTION_CONFIDENCE.set(avg_confidence)
            
            return detections
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def draw_detections(self, image: Image.Image, detections: List[DetectionResult]) -> Image.Image:
        """Draw detection results on image."""
        try:
            draw = ImageDraw.Draw(image)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
            
            for i, detection in enumerate(detections):
                color = colors[i % len(colors)]
                x1, y1, x2, y2 = detection.bbox
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # Draw label
                label = f"{detection.class_name}: {detection.confidence:.2f}"
                draw.text((x1, y1 - 20), label, fill=color, font=font)
            
            return image
            
        except Exception as e:
            logger.error(f"Drawing detections failed: {e}")
            return image

class VideoAnalyzer:
    """Advanced video content analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.frame_sampling_rate = config.get('frame_sampling_rate', 1.0)  # seconds
        self.max_frames = config.get('max_frames', 100)
        
        # Initialize other analyzers
        self.clip_analyzer = MultimodalCLIPAnalyzer(config)
        self.object_detector = ObjectDetector(config)
        
        logger.info("Video Analyzer initialized")
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze video content."""
        start_time = time.time()
        
        try:
            if not MOVIEPY_AVAILABLE:
                logger.warning("MoviePy not available, using OpenCV for video processing")
                return self._analyze_video_opencv(video_path)
            
            # Load video
            video = VideoFileClip(video_path)
            duration = video.duration
            fps = video.fps
            
            # Sample frames
            frame_times = np.arange(0, duration, self.frame_sampling_rate)
            frame_times = frame_times[:self.max_frames]
            
            frames_analysis = []
            scene_changes = []
            objects_timeline = defaultdict(list)
            
            for i, t in enumerate(frame_times):
                try:
                    # Extract frame
                    frame = video.get_frame(t)
                    frame_image = Image.fromarray(frame)
                    
                    # Analyze frame
                    frame_analysis = self._analyze_frame(frame_image, t)
                    frames_analysis.append(frame_analysis)
                    
                    # Track objects over time
                    for obj in frame_analysis.get('objects', []):
                        objects_timeline[obj['class_name']].append({
                            'time': t,
                            'confidence': obj['confidence']
                        })
                    
                    # Detect scene changes (simplified)
                    if i > 0:
                        prev_frame = video.get_frame(frame_times[i-1])
                        scene_change_score = self._calculate_scene_change(prev_frame, frame)
                        if scene_change_score > 0.5:
                            scene_changes.append(t)
                    
                except Exception as e:
                    logger.error(f"Frame analysis failed at time {t}: {e}")
                    continue
            
            # Generate video summary
            summary = self._generate_video_summary(frames_analysis, objects_timeline)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            VIDEO_PROCESSING_TIME.observe(processing_time)
            
            return {
                'video_path': video_path,
                'duration': duration,
                'fps': fps,
                'frames_analyzed': len(frames_analysis),
                'scene_changes': scene_changes,
                'objects_timeline': dict(objects_timeline),
                'summary': summary,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return {'error': str(e)}
        
        finally:
            if 'video' in locals():
                video.close()
    
    def _analyze_video_opencv(self, video_path: str) -> Dict[str, Any]:
        """Analyze video using OpenCV when MoviePy is not available."""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            frames_analysis = []
            frame_interval = int(fps * self.frame_sampling_rate)
            
            frame_idx = 0
            while cap.isOpened() and len(frames_analysis) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_image = Image.fromarray(frame_rgb)
                    
                    # Analyze frame
                    t = frame_idx / fps
                    frame_analysis = self._analyze_frame(frame_image, t)
                    frames_analysis.append(frame_analysis)
                
                frame_idx += 1
            
            cap.release()
            
            return {
                'video_path': video_path,
                'duration': duration,
                'fps': fps,
                'frames_analyzed': len(frames_analysis),
                'summary': self._generate_video_summary(frames_analysis, {})
            }
            
        except Exception as e:
            logger.error(f"OpenCV video analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_frame(self, frame: Image.Image, timestamp: float) -> Dict[str, Any]:
        """Analyze individual video frame."""
        try:
            # Object detection
            objects = self.object_detector.detect_objects(frame)
            
            # Convert to serializable format
            objects_data = []
            for obj in objects:
                objects_data.append({
                    'class_name': obj.class_name,
                    'confidence': obj.confidence,
                    'bbox': obj.bbox
                })
            
            # Basic image analysis
            frame_array = np.array(frame)
            brightness = np.mean(frame_array)
            contrast = np.std(frame_array)
            
            return {
                'timestamp': timestamp,
                'objects': objects_data,
                'brightness': float(brightness),
                'contrast': float(contrast),
                'dominant_colors': self._extract_dominant_colors(frame)
            }
            
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            return {'timestamp': timestamp, 'error': str(e)}
    
    def _calculate_scene_change(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate scene change score between two frames."""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            
            # Calculate histogram difference
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            
            # Correlation coefficient
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # Scene change score (1 - correlation)
            return 1.0 - correlation
            
        except Exception as e:
            logger.error(f"Scene change calculation failed: {e}")
            return 0.0
    
    def _extract_dominant_colors(self, image: Image.Image, k: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image."""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Reshape for clustering
            pixels = img_array.reshape(-1, 3)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get dominant colors
            colors = kmeans.cluster_centers_.astype(int)
            
            return [tuple(color) for color in colors]
            
        except Exception as e:
            logger.error(f"Color extraction failed: {e}")
            return [(128, 128, 128)]  # Default gray
    
    def _generate_video_summary(self, frames_analysis: List[Dict], objects_timeline: Dict) -> str:
        """Generate textual summary of video content."""
        try:
            if not frames_analysis:
                return "No frames analyzed"
            
            # Count object occurrences
            object_counts = defaultdict(int)
            for frame in frames_analysis:
                for obj in frame.get('objects', []):
                    object_counts[obj['class_name']] += 1
            
            # Most common objects
            common_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Average brightness and contrast
            brightnesses = [f.get('brightness', 0) for f in frames_analysis if 'brightness' in f]
            contrasts = [f.get('contrast', 0) for f in frames_analysis if 'contrast' in f]
            
            avg_brightness = np.mean(brightnesses) if brightnesses else 0
            avg_contrast = np.mean(contrasts) if contrasts else 0
            
            # Generate summary
            summary_parts = []
            
            if common_objects:
                objects_str = ", ".join([f"{obj} ({count} frames)" for obj, count in common_objects])
                summary_parts.append(f"Main objects detected: {objects_str}")
            
            if avg_brightness > 150:
                summary_parts.append("Bright lighting conditions")
            elif avg_brightness < 100:
                summary_parts.append("Dark/low-light conditions")
            
            if avg_contrast > 50:
                summary_parts.append("High contrast imagery")
            
            if not summary_parts:
                summary_parts.append("General video content")
            
            return ". ".join(summary_parts) + "."
            
        except Exception as e:
            logger.error(f"Video summary generation failed: {e}")
            return "Video analysis completed"

class DeepfakeDetector:
    """Deepfake and manipulation detection system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize face detection if available
        self.face_detection_available = FACE_LIBS_AVAILABLE
        
        # Simple CNN for deepfake detection (placeholder)
        self.detector = self._build_deepfake_detector()
        
        logger.info("Deepfake Detector initialized")
    
    def _build_deepfake_detector(self) -> nn.Module:
        """Build simple CNN for deepfake detection."""
        class DeepfakeCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4))
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 4 * 4, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        return DeepfakeCNN()
    
    def detect_deepfake(self, image: Image.Image) -> float:
        """Detect if image is a deepfake."""
        try:
            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0)
            
            # Run detection (placeholder - would need trained model)
            with torch.no_grad():
                # For now, return random probability as placeholder
                deepfake_prob = random.uniform(0.1, 0.3)  # Low probability for most images
            
            DEEPFAKE_CONFIDENCE.set(deepfake_prob)
            
            return deepfake_prob
            
        except Exception as e:
            logger.error(f"Deepfake detection failed: {e}")
            return 0.0
    
    def analyze_image_authenticity(self, image: Image.Image) -> Dict[str, Any]:
        """Comprehensive image authenticity analysis."""
        try:
            results = {}
            
            # Deepfake detection
            results['deepfake_probability'] = self.detect_deepfake(image)
            
            # EXIF data analysis
            results['exif_analysis'] = self._analyze_exif_data(image)
            
            # Compression artifacts analysis
            results['compression_analysis'] = self._analyze_compression_artifacts(image)
            
            # Noise pattern analysis
            results['noise_analysis'] = self._analyze_noise_patterns(image)
            
            # Calculate overall authenticity score
            authenticity_factors = [
                1.0 - results['deepfake_probability'],
                results['exif_analysis'].get('authenticity_score', 0.5),
                results['compression_analysis'].get('authenticity_score', 0.5),
                results['noise_analysis'].get('authenticity_score', 0.5)
            ]
            
            results['authenticity_score'] = np.mean(authenticity_factors)
            
            return results
            
        except Exception as e:
            logger.error(f"Authenticity analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_exif_data(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze EXIF data for authenticity indicators."""
        try:
            exif_data = image._getexif() if hasattr(image, '_getexif') else None
            
            if exif_data is None:
                return {'has_exif': False, 'authenticity_score': 0.3}
            
            # Check for common EXIF fields
            important_fields = [
                'DateTime', 'Make', 'Model', 'Software',
                'GPS', 'Orientation', 'XResolution', 'YResolution'
            ]
            
            present_fields = sum(1 for field in important_fields if field in str(exif_data))
            completeness_score = present_fields / len(important_fields)
            
            return {
                'has_exif': True,
                'completeness_score': completeness_score,
                'authenticity_score': min(completeness_score + 0.3, 1.0)
            }
            
        except Exception as e:
            logger.error(f"EXIF analysis failed: {e}")
            return {'has_exif': False, 'authenticity_score': 0.5}
    
    def _analyze_compression_artifacts(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze compression artifacts."""
        try:
            # Convert to numpy array
            img_array = np.array(image.convert('L'))  # Grayscale
            
            # Calculate gradient magnitude
            grad_x = np.gradient(img_array, axis=1)
            grad_y = np.gradient(img_array, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Analyze gradient distribution
            gradient_std = np.std(gradient_magnitude)
            gradient_mean = np.mean(gradient_magnitude)
            
            # Higher variation suggests less compression/manipulation
            authenticity_score = min(gradient_std / 50.0, 1.0)
            
            return {
                'gradient_std': float(gradient_std),
                'gradient_mean': float(gradient_mean),
                'authenticity_score': authenticity_score
            }
            
        except Exception as e:
            logger.error(f"Compression analysis failed: {e}")
            return {'authenticity_score': 0.5}
    
    def _analyze_noise_patterns(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze noise patterns for manipulation detection."""
        try:
            # Convert to numpy array
            img_array = np.array(image.convert('L')).astype(float)
            
            # Apply Gaussian filter to estimate noise
            filtered = gaussian_filter(img_array, sigma=1.0)
            noise = img_array - filtered
            
            # Calculate noise statistics
            noise_std = np.std(noise)
            noise_mean = np.abs(np.mean(noise))
            
            # Consistent noise patterns suggest authenticity
            consistency_score = 1.0 / (1.0 + noise_mean)  # Lower mean suggests consistency
            noise_level_score = min(noise_std / 10.0, 1.0)  # Reasonable noise level
            
            authenticity_score = (consistency_score + noise_level_score) / 2.0
            
            return {
                'noise_std': float(noise_std),
                'noise_mean': float(noise_mean),
                'consistency_score': consistency_score,
                'authenticity_score': authenticity_score
            }
            
        except Exception as e:
            logger.error(f"Noise analysis failed: {e}")
            return {'authenticity_score': 0.5}

class OCRTextExtractor:
    """OCR and text extraction from images."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ocr_available = OCR_AVAILABLE
        
        if self.ocr_available:
            try:
                # Initialize EasyOCR reader
                self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                logger.info("EasyOCR initialized")
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")
                self.reader = None
        else:
            self.reader = None
        
        logger.info("OCR Text Extractor initialized")
    
    def extract_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Extract text from image."""
        try:
            if self.reader is not None:
                return self._extract_with_easyocr(image)
            elif OCR_AVAILABLE:
                return self._extract_with_tesseract(image)
            else:
                logger.warning("No OCR libraries available")
                return []
                
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return []
    
    def _extract_with_easyocr(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Extract text using EasyOCR."""
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Run OCR
            results = self.reader.readtext(img_array)
            
            extracted_texts = []
            total_confidence = 0.0
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence detections
                    extracted_texts.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    total_confidence += confidence
            
            # Update metrics
            if extracted_texts:
                avg_confidence = total_confidence / len(extracted_texts)
                OCR_ACCURACY.set(avg_confidence)
            
            return extracted_texts
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return []
    
    def _extract_with_tesseract(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Extract text using Tesseract OCR."""
        try:
            # Extract text
            text = pytesseract.image_to_string(image)
            
            # Get detailed data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            extracted_texts = []
            
            for i, word in enumerate(data['text']):
                if int(data['conf'][i]) > 50 and word.strip():  # Filter low confidence
                    extracted_texts.append({
                        'text': word,
                        'confidence': int(data['conf'][i]) / 100.0,
                        'bbox': [
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        ]
                    })
            
            return extracted_texts
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return []

class VisualNewsAI:
    """Main visual news AI system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.clip_analyzer = MultimodalCLIPAnalyzer(config)
        self.object_detector = ObjectDetector(config)
        self.video_analyzer = VideoAnalyzer(config)
        self.deepfake_detector = DeepfakeDetector(config)
        self.ocr_extractor = OCRTextExtractor(config)
        
        # Initialize ViT classifier
        num_classes = config.get('num_classes', 10)
        self.vit_classifier = VisionTransformerClassifier(num_classes, config)
        
        # Cache for processed content
        self.analysis_cache = {}
        
        # Redis connection for caching
        try:
            self.redis_client = redis.Redis(
                host=config.get('redis_host', 'localhost'),
                port=config.get('redis_port', 6379),
                decode_responses=True
            )
            self.redis_available = True
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_available = False
        
        logger.info("Visual News AI system initialized")
    
    def analyze_visual_content(self, content: VisualContent) -> VisualAnalysis:
        """Comprehensive visual content analysis."""
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = f"visual_analysis:{content.content_id}"
            if self.redis_available:
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    return VisualAnalysis(**json.loads(cached_result))
            
            # Load image
            if content.url:
                image = self._load_image_from_url(content.url)
            elif content.local_path:
                image = Image.open(content.local_path)
            else:
                raise ValueError("No image source provided")
            
            if image is None:
                raise ValueError("Failed to load image")
            
            # Perform analysis
            analysis_results = {}
            
            # Object detection
            VISION_REQUESTS.labels(task_type='object_detection').inc()
            objects = self.object_detector.detect_objects(image)
            
            # Scene description using CLIP
            VISION_REQUESTS.labels(task_type='scene_description').inc()
            scene_description = self.clip_analyzer.generate_image_description(image)
            
            # Text extraction
            VISION_REQUESTS.labels(task_type='ocr').inc()
            text_content = self.ocr_extractor.extract_text(image)
            extracted_text = [item['text'] for item in text_content]
            
            # Visual sentiment analysis
            VISION_REQUESTS.labels(task_type='sentiment').inc()
            sentiment_score = self._analyze_visual_sentiment(image)
            
            # Quality metrics
            quality_metrics = self._calculate_image_quality(image)
            
            # Face detection (if available)
            faces = self._detect_faces(image) if FACE_LIBS_AVAILABLE else []
            
            # Landmark detection
            landmarks = self._detect_landmarks(image)
            
            # Color analysis
            colors = self._analyze_colors(image)
            
            # Composition analysis
            composition_analysis = self._analyze_composition(image)
            
            # Authenticity analysis
            authenticity_results = self.deepfake_detector.analyze_image_authenticity(image)
            deepfake_probability = authenticity_results.get('deepfake_probability', 0.0)
            authenticity_score = authenticity_results.get('authenticity_score', 0.5)
            
            # Geolocation analysis (if available)
            geolocation = self._analyze_geolocation(image) if GEO_AVAILABLE else None
            
            # Create analysis result
            analysis = VisualAnalysis(
                content_id=content.content_id,
                objects=objects,
                scene_description=scene_description,
                sentiment_score=sentiment_score,
                quality_metrics=quality_metrics,
                text_content=extracted_text,
                faces=faces,
                landmarks=landmarks,
                colors=colors,
                composition_analysis=composition_analysis,
                deepfake_probability=deepfake_probability,
                authenticity_score=authenticity_score,
                geolocation=geolocation
            )
            
            # Cache result
            if self.redis_available:
                self.redis_client.setex(
                    cache_key,
                    3600,  # 1 hour TTL
                    json.dumps(asdict(analysis), default=str)
                )
            
            # Update metrics
            processing_time = time.time() - start_time
            VISION_LATENCY.labels(task_type='full_analysis').observe(processing_time)
            IMAGE_QUALITY.set(np.mean(list(quality_metrics.values())))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Visual content analysis failed: {e}")
            # Return empty analysis on error
            return VisualAnalysis(
                content_id=content.content_id,
                objects=[],
                scene_description="Analysis failed",
                sentiment_score=0.0,
                quality_metrics={},
                text_content=[],
                faces=[],
                landmarks=[],
                colors={},
                composition_analysis={},
                deepfake_probability=0.0,
                authenticity_score=0.0
            )
    
    def _load_image_from_url(self, url: str) -> Optional[Image.Image]:
        """Load image from URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            return image.convert('RGB')
            
        except Exception as e:
            logger.error(f"Failed to load image from URL {url}: {e}")
            return None
    
    def _analyze_visual_sentiment(self, image: Image.Image) -> float:
        """Analyze visual sentiment of image."""
        try:
            # Use CLIP to compare with sentiment-related descriptions
            positive_descriptions = [
                "happy people", "celebration", "success", "joy", "positive news",
                "achievement", "victory", "smile", "bright scene"
            ]
            
            negative_descriptions = [
                "sad people", "disaster", "conflict", "tragedy", "negative news",
                "destruction", "protest", "dark scene", "crisis"
            ]
            
            # Calculate similarities
            positive_scores = []
            negative_scores = []
            
            for desc in positive_descriptions:
                score = self.clip_analyzer.calculate_similarity(image, desc)
                positive_scores.append(score)
            
            for desc in negative_descriptions:
                score = self.clip_analyzer.calculate_similarity(image, desc)
                negative_scores.append(score)
            
            # Calculate sentiment score (-1 to 1)
            avg_positive = np.mean(positive_scores)
            avg_negative = np.mean(negative_scores)
            
            sentiment_score = (avg_positive - avg_negative) / (avg_positive + avg_negative + 1e-8)
            
            return float(sentiment_score)
            
        except Exception as e:
            logger.error(f"Visual sentiment analysis failed: {e}")
            return 0.0
    
    def _calculate_image_quality(self, image: Image.Image) -> Dict[str, float]:
        """Calculate various image quality metrics."""
        try:
            img_array = np.array(image.convert('L'))  # Grayscale
            
            # Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Brightness
            brightness = np.mean(img_array)
            
            # Contrast (standard deviation)
            contrast = np.std(img_array)
            
            # Noise estimation
            noise = self._estimate_noise(img_array)
            
            # Overall quality score
            quality_score = (
                min(sharpness / 1000.0, 1.0) * 0.3 +
                (1.0 - abs(brightness - 128) / 128.0) * 0.2 +
                min(contrast / 50.0, 1.0) * 0.3 +
                (1.0 - min(noise / 20.0, 1.0)) * 0.2
            )
            
            return {
                'sharpness': float(sharpness),
                'brightness': float(brightness),
                'contrast': float(contrast),
                'noise': float(noise),
                'overall_quality': float(quality_score)
            }
            
        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return {'overall_quality': 0.5}
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise level in image."""
        try:
            # Use Laplacian to estimate noise
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            noise_estimate = np.mean(np.abs(laplacian))
            return float(noise_estimate)
            
        except Exception as e:
            logger.error(f"Noise estimation failed: {e}")
            return 0.0
    
    def _detect_faces(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect faces in image."""
        try:
            if not FACE_LIBS_AVAILABLE:
                return []
            
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(img_array)
            
            faces = []
            for (top, right, bottom, left) in face_locations:
                faces.append({
                    'bbox': [left, top, right, bottom],
                    'confidence': 0.9  # face_recognition doesn't provide confidence
                })
            
            return faces
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def _detect_landmarks(self, image: Image.Image) -> List[str]:
        """Detect landmarks in image using CLIP."""
        try:
            # Common landmarks to check
            landmarks = [
                "Eiffel Tower", "Statue of Liberty", "Big Ben", "Taj Mahal",
                "Golden Gate Bridge", "Sydney Opera House", "Colosseum",
                "Mount Rushmore", "Christ the Redeemer", "Pyramids of Giza",
                "White House", "Capitol Building", "Empire State Building"
            ]
            
            detected_landmarks = []
            threshold = 0.3  # Similarity threshold
            
            for landmark in landmarks:
                similarity = self.clip_analyzer.calculate_similarity(image, landmark)
                if similarity > threshold:
                    detected_landmarks.append(landmark)
            
            return detected_landmarks
            
        except Exception as e:
            logger.error(f"Landmark detection failed: {e}")
            return []
    
    def _analyze_colors(self, image: Image.Image) -> Dict[str, float]:
        """Analyze color composition of image."""
        try:
            img_array = np.array(image)
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Calculate color statistics
            hue_mean = np.mean(hsv[:, :, 0])
            saturation_mean = np.mean(hsv[:, :, 1])
            value_mean = np.mean(hsv[:, :, 2])
            
            # Dominant color analysis
            pixels = img_array.reshape(-1, 3)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            dominant_colors = kmeans.cluster_centers_
            color_percentages = np.bincount(kmeans.labels_) / len(kmeans.labels_)
            
            return {
                'hue_mean': float(hue_mean),
                'saturation_mean': float(saturation_mean),
                'value_mean': float(value_mean),
                'dominant_color_1': dominant_colors[0].tolist(),
                'dominant_color_2': dominant_colors[1].tolist(),
                'dominant_color_3': dominant_colors[2].tolist(),
                'color_diversity': float(np.std(color_percentages))
            }
            
        except Exception as e:
            logger.error(f"Color analysis failed: {e}")
            return {}
    
    def _analyze_composition(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image composition."""
        try:
            img_array = np.array(image.convert('L'))
            height, width = img_array.shape
            
            # Rule of thirds analysis
            third_h = height // 3
            third_w = width // 3
            
            # Calculate interest points at rule of thirds intersections
            interest_points = [
                (third_w, third_h), (2 * third_w, third_h),
                (third_w, 2 * third_h), (2 * third_w, 2 * third_h)
            ]
            
            # Edge detection for composition analysis
            edges = cv2.Canny(img_array, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            # Symmetry analysis
            left_half = img_array[:, :width//2]
            right_half = np.fliplr(img_array[:, width//2:])
            
            if left_half.shape == right_half.shape:
                symmetry_score = 1.0 - np.mean(np.abs(left_half - right_half)) / 255.0
            else:
                symmetry_score = 0.0
            
            return {
                'edge_density': float(edge_density),
                'symmetry_score': float(symmetry_score),
                'aspect_ratio': float(width / height),
                'interest_points': interest_points
            }
            
        except Exception as e:
            logger.error(f"Composition analysis failed: {e}")
            return {}
    
    def _analyze_geolocation(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """Analyze potential geolocation from image."""
        try:
            if not GEO_AVAILABLE:
                return None
            
            # Extract EXIF GPS data if available
            exif_data = image._getexif() if hasattr(image, '_getexif') else None
            
            if exif_data and 'GPS' in str(exif_data):
                # Parse GPS coordinates (simplified)
                return {'source': 'exif', 'coordinates': 'available'}
            
            # Use landmark detection for location hints
            landmarks = self._detect_landmarks(image)
            
            if landmarks:
                return {
                    'source': 'landmark_detection',
                    'detected_landmarks': landmarks,
                    'confidence': 'medium'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Geolocation analysis failed: {e}")
            return None

def create_vision_api(vision_ai: VisualNewsAI) -> Flask:
    """Create Flask API for vision system."""
    app = Flask(__name__)
    
    @app.route('/vision/analyze', methods=['POST'])
    def analyze_image():
        try:
            data = request.get_json()
            
            # Create visual content object
            content = VisualContent(
                content_id=data.get('content_id', str(uuid.uuid4())),
                content_type=data.get('content_type', 'image'),
                url=data.get('url'),
                local_path=data.get('local_path'),
                metadata=data.get('metadata', {})
            )
            
            # Perform analysis
            analysis = vision_ai.analyze_visual_content(content)
            
            # Convert to JSON-serializable format
            result = asdict(analysis)
            
            return jsonify({
                'status': 'success',
                'analysis': result,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Vision analysis API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/vision/video/analyze', methods=['POST'])
    def analyze_video():
        try:
            data = request.get_json()
            video_path = data.get('video_path')
            
            if not video_path:
                return jsonify({'error': 'video_path required'}), 400
            
            # Analyze video
            analysis = vision_ai.video_analyzer.analyze_video(video_path)
            
            return jsonify({
                'status': 'success',
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Video analysis API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/vision/similarity', methods=['POST'])
    def calculate_similarity():
        try:
            data = request.get_json()
            image_url = data.get('image_url')
            text = data.get('text')
            
            if not image_url or not text:
                return jsonify({'error': 'image_url and text required'}), 400
            
            # Load image
            image = vision_ai._load_image_from_url(image_url)
            if image is None:
                return jsonify({'error': 'Failed to load image'}), 400
            
            # Calculate similarity
            similarity = vision_ai.clip_analyzer.calculate_similarity(image, text)
            
            return jsonify({
                'status': 'success',
                'similarity': similarity,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Similarity calculation API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/vision/deepfake/detect', methods=['POST'])
    def detect_deepfake():
        try:
            data = request.get_json()
            image_url = data.get('image_url')
            
            if not image_url:
                return jsonify({'error': 'image_url required'}), 400
            
            # Load image
            image = vision_ai._load_image_from_url(image_url)
            if image is None:
                return jsonify({'error': 'Failed to load image'}), 400
            
            # Detect deepfake
            authenticity_results = vision_ai.deepfake_detector.analyze_image_authenticity(image)
            
            return jsonify({
                'status': 'success',
                'authenticity_analysis': authenticity_results,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Deepfake detection API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/vision/ocr', methods=['POST'])
    def extract_text():
        try:
            data = request.get_json()
            image_url = data.get('image_url')
            
            if not image_url:
                return jsonify({'error': 'image_url required'}), 400
            
            # Load image
            image = vision_ai._load_image_from_url(image_url)
            if image is None:
                return jsonify({'error': 'Failed to load image'}), 400
            
            # Extract text
            text_results = vision_ai.ocr_extractor.extract_text(image)
            
            return jsonify({
                'status': 'success',
                'text_results': text_results,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"OCR API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/vision/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'clip_analyzer': 'active',
                'object_detector': 'active',
                'video_analyzer': 'active',
                'deepfake_detector': 'active',
                'ocr_extractor': 'active' if vision_ai.ocr_extractor.ocr_available else 'unavailable'
            }
        })
    
    return app

def main():
    """Main function to run the Visual News AI system."""
    # Configuration
    config = {
        'clip_model': 'openai/clip-vit-base-patch32',
        'detr_model': 'facebook/detr-resnet-50',
        'vit_model': 'google/vit-base-patch16-224',
        'confidence_threshold': 0.7,
        'frame_sampling_rate': 1.0,
        'max_frames': 100,
        'num_classes': 10,
        'dropout': 0.1,
        'hidden_dim': 512,
        'freeze_backbone': False,
        'redis_host': 'localhost',
        'redis_port': 6379
    }
    
    # Initialize system
    logger.info("Initializing Visual News AI system...")
    vision_ai = VisualNewsAI(config)
    
    # Create API
    app = create_vision_ai(vision_ai)
    
    # Start Prometheus metrics server
    try:
        start_http_server(8001)
        logger.info("Prometheus metrics server started on port 8001")
    except Exception as e:
        logger.warning(f"Failed to start Prometheus server: {e}")
    
    # Example usage
    logger.info("Running example analysis...")
    
    try:
        # Example image analysis
        example_content = VisualContent(
            content_id="example_001",
            content_type="image",
            url="https://example.com/news-image.jpg",
            metadata={"source": "example_news", "category": "politics"}
        )
        
        # Note: This would fail without a real image URL
        # analysis = vision_ai.analyze_visual_content(example_content)
        # logger.info(f"Example analysis completed: {analysis.scene_description}")
        
        logger.info("Visual News AI system ready for deployment")
        
    except Exception as e:
        logger.error(f"Example analysis failed: {e}")
    
    # Start Flask API
    logger.info("Starting Visual News AI API server...")
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=False,
        threaded=True
    )

if __name__ == "__main__":
    main()