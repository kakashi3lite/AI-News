# Computer Vision Module

A comprehensive computer vision system for analyzing visual content in news media, including images and videos.

## Features

### Core Capabilities
- **Object Detection**: Identify and locate objects in images using DETR models
- **Scene Description**: Generate natural language descriptions using CLIP
- **Text Extraction**: OCR capabilities with EasyOCR and Tesseract
- **Sentiment Analysis**: Visual sentiment analysis using multimodal models
- **Quality Assessment**: Image quality metrics (sharpness, brightness, contrast, noise)
- **Face Detection**: Detect and analyze faces in images
- **Landmark Recognition**: Identify famous landmarks and locations
- **Color Analysis**: Analyze color composition and dominant colors
- **Composition Analysis**: Rule of thirds, symmetry, and edge detection
- **Authenticity Detection**: Deepfake and manipulation detection
- **Video Analysis**: Frame extraction and temporal analysis
- **Geolocation**: Extract location data from EXIF and visual cues

### Advanced Features
- **Caching**: Redis-based result caching for performance
- **Monitoring**: Prometheus metrics integration
- **Graceful Degradation**: Handles missing optional dependencies
- **REST API**: Flask-based web service
- **Batch Processing**: Efficient processing of multiple images

## Installation

### Basic Installation
```bash
pip install -r requirements.txt
```

### Optional Dependencies
Some features require additional dependencies:

```bash
# For video processing
pip install moviepy

# For face detection
pip install face-recognition dlib

# For OCR
pip install pytesseract easyocr

# For geospatial analysis
pip install geopandas folium geopy
```

## Usage

### Python API

```python
from visual_news_ai import VisualNewsAI

# Initialize the system
vision_ai = VisualNewsAI()

# Analyze an image
result = vision_ai.analyze_visual_content(
    image_url="https://example.com/image.jpg"
)

print(f"Scene: {result.scene_description}")
print(f"Objects: {result.objects}")
print(f"Sentiment: {result.sentiment_score}")
print(f"Text: {result.text_content}")
```

### REST API

```bash
# Start the API server
python visual_news_ai.py
```

The API will be available at `http://localhost:5002`

#### Endpoints

- `POST /vision/analyze` - Analyze image content
- `POST /vision/video/analyze` - Analyze video content
- `POST /vision/similarity` - Calculate image-text similarity
- `POST /vision/deepfake` - Detect deepfakes
- `POST /vision/ocr` - Extract text from images
- `GET /vision/health` - Health check

#### Example Request

```bash
curl -X POST http://localhost:5002/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg"}'
```

## Configuration

The system can be configured through environment variables:

```bash
# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Model configuration
CLIP_MODEL=openai/clip-vit-base-patch32
DETR_MODEL=facebook/detr-resnet-50
VIT_MODEL=google/vit-base-patch16-224

# API configuration
API_HOST=0.0.0.0
API_PORT=5002

# Monitoring
PROMETHEUS_PORT=8000
```

## Architecture

### Core Components

1. **VisionTransformerClassifier**: Image classification using Vision Transformers
2. **MultimodalCLIPAnalyzer**: Scene understanding and similarity computation
3. **ObjectDetector**: Object detection and localization
4. **VideoAnalyzer**: Video processing and frame analysis
5. **DeepfakeDetector**: Authenticity and manipulation detection
6. **OCRTextExtractor**: Text extraction from images
7. **NoiseAnalyzer**: Image quality and noise analysis

### Data Flow

1. **Input Processing**: Load and preprocess images/videos
2. **Feature Extraction**: Extract visual features using deep learning models
3. **Analysis Pipeline**: Run multiple analysis components in parallel
4. **Result Aggregation**: Combine results into comprehensive analysis
5. **Caching**: Store results for future requests
6. **Response**: Return structured analysis results

## Performance

### Optimization Features
- **Model Caching**: Pre-loaded models for faster inference
- **Result Caching**: Redis-based caching of analysis results
- **Batch Processing**: Efficient processing of multiple items
- **Graceful Degradation**: Continues working with missing dependencies
- **Memory Management**: Efficient memory usage for large images

### Monitoring

The system includes comprehensive monitoring:
- **Request Metrics**: Track API usage and performance
- **Processing Time**: Monitor analysis duration
- **Error Rates**: Track failures and exceptions
- **Resource Usage**: Monitor memory and CPU usage

## Error Handling

The system includes robust error handling:
- **Dependency Checks**: Gracefully handles missing optional dependencies
- **Network Errors**: Retries and fallbacks for network requests
- **Model Errors**: Fallback strategies for model failures
- **Input Validation**: Comprehensive input validation
- **Logging**: Detailed logging for debugging

## Development

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=visual_news_ai tests/
```

### Code Quality

```bash
# Format code
black visual_news_ai.py

# Lint code
flake8 visual_news_ai.py

# Type checking
mypy visual_news_ai.py
```

## License

This project is part of the AI News Dashboard system.

## Contributing

Please refer to the main project's CONTRIBUTING.md for contribution guidelines.