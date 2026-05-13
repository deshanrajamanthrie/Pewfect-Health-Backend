# Dog Emotion Detection AI Service

A machine learning service that analyzes dog barks to detect emotional states. Uses deep learning to classify emotions from audio features.

## Overview

This Python-based service processes dog bark audio files and uses a trained neural network model to predict emotional states. It's designed to integrate with the Pawfect Health backend API.

## Features

- Dog bark emotion classification
- Audio feature extraction using librosa
- Pre-trained TensorFlow/Keras model
- Easy-to-integrate REST API endpoints
- Support for model retraining with new data

## Requirements

- Python 3.8 or higher
- librosa (audio processing library)
- requests (HTTP client)
- TensorFlow/Keras (deep learning)
- numpy, pandas (data processing)

## Installation

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
python -m pip install requests
python -m pip install librosa
python -m pip install tensorflow
python -m pip install keras
python -m pip install numpy pandas
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Service

```bash
python app.py
```

### Training the Model

Retrain the emotion model with your dataset:

```bash
python train_emotion_model.py
```

Or use the general training script:

```bash
python train_model.py
```

## Project Structure

```
ai_service/
├── app.py                      # Main application (Flask/FastAPI)
├── train_emotion_model.py      # Emotion model training script
├── train_model.py              # General model training
├── dog_emotion_model.h5        # Pre-trained model (Keras/H5 format)
├── data/
│   ├── dog_bark_emotions.csv   # Basic emotion dataset
│   ├── dog_bark_emotions_large.csv # Extended dataset
│   └── dog_data.csv            # General dog data
├── model/                      # Saved models directory
└── readme.md                   # This file
```

## Model Details

### Input

- Audio files (.wav, .mp3, etc.)
- Features extracted: MFCC, spectral centroids, etc.

### Output

- Emotional classification (e.g., happy, sad, aggressive, anxious)
- Confidence scores

### Model File

- **dog_emotion_model.h5** - Keras model in HDF5 format

## Data Files

- **dog_bark_emotions.csv** - Contains bark samples with emotion labels
- **dog_bark_emotions_large.csv** - Extended dataset for improved accuracy
- **dog_data.csv** - General dog-related information

## Configuration

Edit the following in `app.py` as needed:

```python
MODEL_PATH = './dog_emotion_model.h5'
API_PORT = 5000
DEBUG = True
```

## API Integration

### Example Request (from NestJS backend)

```python
import requests

# Send audio for emotion prediction
with open('dog_bark.wav', 'rb') as f:
    files = {'audio': f}
    response = requests.post('http://localhost:5000/predict', files=files)
    emotion = response.json()
    print(emotion)  # Output: {'emotion': 'happy', 'confidence': 0.95}
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution:** Ensure virtual environment is activated:

```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows
```

### Issue: Model file not found

**Solution:** Verify `dog_emotion_model.h5` is in the same directory as `app.py`

### Issue: Audio processing error

**Solution:** Install librosa audio dependencies:

```bash
pip install --upgrade librosa
```

### Issue: TensorFlow GPU errors

**Solution:** Use CPU version:

```bash
pip uninstall tensorflow-gpu
pip install tensorflow
```

## Dependencies Summary

| Package    | Purpose                                   |
| ---------- | ----------------------------------------- |
| librosa    | Audio file loading and feature extraction |
| requests   | HTTP requests to backend                  |
| tensorflow | Deep learning framework                   |
| keras      | Model training and inference              |
| numpy      | Numerical computations                    |
| pandas     | Data manipulation                         |

## Next Steps

1. Ensure virtual environment is activated
2. Install all dependencies
3. Run `python app.py` to start the service
4. The NestJS backend can now make requests to this service

## Performance Tips

- Use GPU acceleration if available (CUDA for TensorFlow)
- Pre-process audio files to consistent sample rates
- Implement caching for frequently analyzed audio
