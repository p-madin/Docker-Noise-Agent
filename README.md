# AI Noise Classifier

Real-time environmental sound classification using deep learning. Detects 58 different sound categories including animals, vehicles, household sounds, and more.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/p-madin/Docker-Noise-Agent
cd AI-Noise

# Start with Docker Compose
docker compose up

# Visit the web interface
# http://localhost:80
```

The first startup will automatically download the required datasets (~6GB total).

## Overview

This project provides a real-time audio classification system with:
- **Pre-trained CNN Model**: 92% accuracy on 58 sound categories
- **Web Interface**: Live microphone monitoring with detection history
- **Training Pipeline**: Optimized GPU training (<1 hour on consumer hardware)
- **Database Persistence**: MySQL backend for detection logging and HITL flagging

### Architecture

- **Backend**: Flask server with TensorFlow/Keras inference
- **Frontend**: Vanilla JavaScript with real-time audio capture
- **Training**: TFRecord pipeline with GPU-accelerated SpecAugment
- **Datasets**: ESC-50 + UrbanSound8K (auto-downloaded)

## Retraining the Model

To retrain from scratch or fine-tune:

```bash
# Inside the container
docker exec -it ai-noise-app-1 bash
cd /root/Workspace
python compile_classifiers.py
```

**Training Time**: ~1 hour on NVIDIA consumer GPU  
**Output**: `noise_classifier_cnn.h5`, `label_encoder.joblib`

### Training Configuration

Edit `compile_classifiers.py` to adjust:
- `EPOCHS = 30` - Number of training epochs
- `BATCH_SIZE = 64` - Batch size (adjust for GPU memory)
- `N_MELS = 128` - Mel spectrogram resolution
- Data augmentation parameters (SpecAugment time/frequency masking)

## Bundled Utilities

### `validate_classifier.py`
**Purpose**: Test model accuracy on random samples from the dataset.

```bash
python validate_classifier.py
```

**Output**: Accuracy metrics, top-3 predictions, and per-class performance.

### `noise_resistance_test.py`
**Purpose**: Evaluate model robustness against varying noise levels (SNR).

```bash
python noise_resistance_test.py
```

**Output**: Accuracy degradation curve from clean audio (30dB) to heavy noise (0dB).

### `benchmark_suite.py`
**Purpose**: Measure end-to-end training time and epoch performance.

```bash
python benchmark_suite.py
```

**Output**: Total training duration, average epoch time, and accuracy metrics.  
**Note**: Clears checkpoints for consistent benchmarking.

### `list_categories.py`
**Purpose**: Display all 58 trained sound categories.

```bash
python list_categories.py
```

### `analyze_dataset.py`
**Purpose**: Generate dataset statistics (class distribution, file counts).

```bash
python analyze_dataset.py
```

### `generate_ai_dreams.py`
**Purpose**: Synthesize audio samples using gradient ascent on the trained model.

```bash
python generate_ai_dreams.py
```

**Output**: Generated audio files in `dreams/` directory.

### `inspect_classes.py`
**Purpose**: Visualize mel spectrograms for specific sound classes.

```bash
python inspect_classes.py
```

## Web Interface Features

- **Live Monitoring**: Real-time microphone classification
- **Detection History**: Persistent log of all detections (last 50)
- **Audio Playback**: Review recorded samples
- **HITL Flagging**: Mark incorrect predictions for retraining

### Adjusting Confidence Threshold

Edit `app/audio_engine.py`:

```python
if confidence >= 0.95:  # Default: 95% confidence required
    return label, float(confidence)
```

Lower values increase sensitivity but may introduce false positives.

## Performance Benchmarks

| Pipeline Version | Training Time | GPU Utilization | Accuracy |
|-----------------|---------------|-----------------|----------|
| Generator (CPU) | ~4-5 hours    | 3-7%           | N/A      |
| In-Memory       | Crashed (OOM) | N/A            | N/A      |
| **TFRecord**    | **<1 hour**   | **~90%**       | **92%**  |

See `app/compiler benchmark/framework versions.md` for detailed analysis.

## Project Structure

```
AI-Noise/
├── app/
│   ├── server.py              # Flask web server
│   ├── audio_engine.py        # CNN inference engine
│   ├── compile_classifiers.py # Training pipeline
│   ├── static/                # CSS/JS for web UI
│   ├── templates/             # HTML templates
│   └── datasets/              # Auto-downloaded (gitignored)
├── Dockerfile                 # Container definition
├── compose.yaml              # Docker Compose config
├── startup.sh                # Dataset provisioning + server start
└── requirements.txt          # Python dependencies
```

## License

See [LICENSE](LICENSE) file for details.

## Datasets

- **ESC-50**: Environmental Sound Classification (2,000 samples, 50 classes)
- **UrbanSound8K**: Urban sound recordings (8,732 samples, 10 classes)

Both datasets are automatically downloaded on first startup.
