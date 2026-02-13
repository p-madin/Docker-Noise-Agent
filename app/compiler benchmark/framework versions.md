# AI Noise Classifier - Framework Optimization Journey

This document chronicles the optimization attempts to improve training speed and GPU utilization for the AI Noise Classifier.

## 1. Baseline: Generator Pipeline (CPU-Bound)
**Method**: Used `librosa.load` and `librosa.effects.pitch_shift` inside a Python generator, feeding `model.fit`.
**Hardware Suitability**: 
- Single-core CPU systems where GPU speed is negligible.
- Extremely large datasets that physically cannot fit in storage (streaming only).
**Why it failed (Performance)**:
- **GPU Starvation**: GPU utilization hovered at ~3-7%.
- **CPU Bottleneck**: Decoding audio and performing FFT/Shift on the fly is computationally expensive. The GPU spent 93% of its time waiting for the CPU.
**Training Time**: Estimated 4-5 hours.

## 2. Approach: In-Memory Loading (OOM Crash)
**Method**: Pre-loaded all 52,000+ audio files (original + augmented versions) into a massive NumPy array in RAM before training.
**Hardware Suitability**:
- Systems with massive amounts of RAM (>64GB).
- Very small datasets (e.g., <5,000 samples).
**Why it failed**:
- **OOM (Out of Memory)**: The uncompressed float32 arrays for 150k+ spectrograms consumed >12GB of RAM, causing the OS to kill the process.
**Training Time**: N/A (Crashed).

## 3. Solution: TFRecord Pipeline + GPU Augmentation
**Method**: 
1. **Pre-computation**: Converted all audio to Mel Spectrograms once and saved as linear `TFRecord` files (binary format).
2. **GPU Augmentation**: Moved `pitch_shift` (CPU) to `SpecAugment` (GPU/TensorFlow ops) using Time/Frequency masking.
**Hardware Suitability**:
- Standard Deep Learning setups (Consumer GPUs).
- Limited RAM environments (streaming from disk).
**Result**:
- **GPU Utilization**: ~30-50%+ (Waiting for data is minimized).
- **Training Time**: < 1 Hour.
- **Accuracy**: ~92%.

## 4. Future Advancements

### Divide and Conquer Mechanisms
- **Sharding**: Split TFRecords into multiple shards (e.g., `train-001.tfrecord`, `train-002.tfrecord`) to allow parallel reading from multiple disk sectors.
- **Multi-Worker Strategy**: Use `tf.distribute.MirroredStrategy` to split the batch across multiple GPUs if available.

### Advanced Data Loading
- **DALI (NVIDIA Data Loading Library)**: Move the JPEG/Audio decoding itself to the GPU (if supported).
- **Interleaved Reading**: `dataset.interleave` to read from multiple TFRecord files simultaneously, further saturating I/O.

### Model Architecture
- **EfficientNet / MobileNet**: Switch to lighter backbones if inference speed becomes a bottleneck.
- **Distillation**: Train a massive model (Teacher) and distill knowledge into a tiny model (Student) for edge deployment.
