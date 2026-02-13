import os
import shutil
import json
from datetime import datetime
import librosa
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Configuration
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
DATASET_PATH = 'datasets'
MODEL_OUTPUT = 'noise_classifier_cnn.h5'
ENCODER_OUTPUT = 'label_encoder.joblib'
BACKUP_DIR = 'backups'
CHECKPOINT_DIR = 'checkpoints'
SAMPLE_RATE = 16000
DURATION = 1.0
N_MELS = 128
FIXED_WIDTH = 128

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

# --- Feature Extraction (Same as inference) ---

def extract_mel_spectrogram(audio, sample_rate):
    try:
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio = audio / max_amp

        target_len = int(sample_rate * DURATION)
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, target_len - len(audio)))

        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=N_MELS, n_fft=2048, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=1.0) 
        
        if mel_db.shape[1] > FIXED_WIDTH:
            mel_db = mel_db[:, :FIXED_WIDTH]
        else:
            mel_db = np.pad(mel_db, ((0, 0), (0, FIXED_WIDTH - mel_db.shape[1])))
            
        return mel_db[..., np.newaxis]
    except:
        return None

# --- Data Augmentation ---

# --- Data Augmentation (GPU) ---

def spec_augment(spectrogram, label):
    """
    Applies SpecAugment (Time/Freq masking) to a spectrogram.
    Runs on GPU as part of the tf.data pipeline.
    """
    # 1. Frequency Masking
    # Mask up to n_freq_mask bands
    n_freq_mask = 2
    freq_mask_param = 10
    
    # 2. Time Masking
    # Mask up to n_time_mask blocks
    n_time_mask = 2
    time_mask_param = 10
    
    spectrogram_aug = spectrogram
    
    # Add noise (Gaussian) - GPU efficient
    noise = tf.random.normal(tf.shape(spectrogram), mean=0.0, stddev=0.05)
    spectrogram_aug = spectrogram_aug + noise

    # Frequency masking
    for _ in range(n_freq_mask):
        f = tf.random.uniform([], minval=0, maxval=freq_mask_param, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=N_MELS - f, dtype=tf.int32)
        
        # Create mask
        # shape: (1, N_MELS, 1, 1) to broadcast? 
        # Easier to just zero out rows
        # We need to construct indices or use flexible masking
        # TF way:
        mask_start = f0
        mask_end = f0 + f
        
        # Creating a mask of ones and setting [f0:f0+f, :] to 0 is surprisingly hard in pure TF graph mode without loops
        # Alternative: tensorflow_io has spec_augment, but we want pure TF.
        # Simple Rectangular Masking logic:
        
        # Grid of coordinates
        h = tf.range(N_MELS)
        w = tf.range(FIXED_WIDTH)
        
        mask_h = (h >= mask_start) & (h < mask_end)
        mask_h = tf.reshape(mask_h, [N_MELS, 1, 1])
        mask_h = tf.cast(mask_h, tf.float32)
        
        # Invert mask (1 where we keep, 0 where we mask)
        keep_mask = 1.0 - mask_h
        spectrogram_aug = spectrogram_aug * keep_mask

    # Time masking
    for _ in range(n_time_mask):
        t = tf.random.uniform([], minval=0, maxval=time_mask_param, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=FIXED_WIDTH - t, dtype=tf.int32)
        
        mask_start = t0
        mask_end = t0 + t
        
        w_idx = tf.range(FIXED_WIDTH)
        mask_w = (w_idx >= mask_start) & (w_idx < mask_end)
        mask_w = tf.reshape(mask_w, [1, FIXED_WIDTH, 1])
        mask_w = tf.cast(mask_w, tf.float32)
        
        keep_mask = 1.0 - mask_w
        spectrogram_aug = spectrogram_aug * keep_mask
        
    return spectrogram_aug, label

def get_tfrecord_dataset(tfrecord_path, augment=False, batch_size=64):
    """Creates a highly optimized dataset from TFRecords."""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # 1. Parse
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 2. Cache (if small enough) or Shuffle
    # For 64k samples * 128*128*4bytes ~ 4GB. Might fit in RAM if we have 16GB.
    # Safe bet: Shuffle buffer
    dataset = dataset.shuffle(buffer_size=5000)
    
    # 3. Augment (GPU)
    if augment:
        dataset = dataset.map(spec_augment, num_parallel_calls=tf.data.AUTOTUNE)
        
    # 4. Batch & Prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# --- Streaming Pipeline ---

# --- TFRecord Pipeline ---

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(spectrogram, label):
    """
    Creates a tf.train.Example message ready to be written to a file.
    spectrogram: Numpy array (128, 128, 1) float32
    label: Integer
    """
    feature = {
        'spectrogram': _bytes_feature(spectrogram.tobytes()),
        'label': _int64_feature(label),
        'shape': _int64_feature(spectrogram.shape[0]), # Save height
        'width': _int64_feature(spectrogram.shape[1]), # Save width
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def parse_tfrecord(example_proto):
    """Parses a single TFRecord string into tensors."""
    feature_description = {
        'spectrogram': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'shape': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
    }
    features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode spectrogram
    spectrogram = tf.io.decode_raw(features['spectrogram'], tf.float32)
    
    # Reshape based on fixed constants or parsed shape if implemented fully dynamic
    # Here we assume fixed constants for simplicity and speed
    spectrogram = tf.reshape(spectrogram, (N_MELS, FIXED_WIDTH, 1))
    
    label = tf.cast(features['label'], tf.int32)
    
    return spectrogram, label

def save_as_tfrecords(file_list, label_list, filename="train.tfrecord"):
    """
    Converts audio files to Mel Spectrograms and saves them as a TFRecord file.
    This runs ONCE and enables extremely fast training later.
    """
    if os.path.exists(filename):
        print(f"TFRecord {filename} already exists. Skipping generation.")
        return

    print(f"Generating {filename} from {len(file_list)} files...")
    
    with tf.io.TFRecordWriter(filename) as writer:
        for i, (file_path, label) in enumerate(zip(file_list, label_list)):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(file_list)}...", end='\r')
            
            try:
                # 1. Load Audio
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                seg_len = int(sr * DURATION)
                
                # 2. Extract Segment (First 1s)
                segment = audio[:seg_len]
                
                # Skip overly short files
                if len(segment) < seg_len // 2:
                    continue
                
                # Pad if needed
                if len(segment) < seg_len:
                    segment = np.pad(segment, (0, seg_len - len(segment)))
                
                # 3. Compute Mel Spectrum (No augmentation here!)
                mel_spec = extract_mel_spectrogram(segment, sr)
                
                if mel_spec is not None:
                    # Serialize and Write
                    example = serialize_example(mel_spec.astype(np.float32), label)
                    writer.write(example)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
                
    print(f"\nSuccessfully created {filename}.")

class LogCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1} finished. Logs: {logs}")

    def on_test_begin(self, logs=None):
        print("Starting validation...")

    def on_test_end(self, logs=None):
        print("Validation finished.")

# --- Model Arch ---

def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((4, 4)), # Stronger pooling for high res
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Backup Logic ---

def backup_current_model():
    """Back up the existing model and encoder if they exist."""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        print(f"Created backup directory: {BACKUP_DIR}")
    
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"Created checkpoint directory: {CHECKPOINT_DIR}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for file_path in [MODEL_OUTPUT, ENCODER_OUTPUT]:
        if os.path.exists(file_path):
            base, ext = os.path.splitext(file_path)
            backup_name = f"{base}_{timestamp}{ext}"
            backup_path = os.path.join(BACKUP_DIR, backup_name)
            shutil.copy2(file_path, backup_path)
            print(f"Backed up {file_path} to {backup_path}")

# --- Configuration Management ---

def save_training_config():
    """Save current training configuration for validation on resume."""
    config = {
        'sample_rate': SAMPLE_RATE,
        'duration': DURATION,
        'n_mels': N_MELS,
        'fixed_width': FIXED_WIDTH,
        'epochs': 30,
        'steps_per_epoch': 1000,
        'validation_steps': 100,
        'batch_size': 64,  # Increased for GPU utilization
        'data_version': 2  # Version 2: Fixed segmentation (one sample per file)
    }
    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    config_path = os.path.join(CHECKPOINT_DIR, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return config

def validate_training_config():
    """Check if current config matches saved config from checkpoint."""
    config_path = os.path.join(CHECKPOINT_DIR, 'training_config.json')
    if not os.path.exists(config_path):
        return True, None  # No config file, assume first run
    
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
    
    current_config = {
        'sample_rate': SAMPLE_RATE,
        'duration': DURATION,
        'n_mels': N_MELS,
        'fixed_width': FIXED_WIDTH,
        'epochs': 30,
        'steps_per_epoch': 1000,
        'validation_steps': 100,
        'batch_size': 64,  # Increased for GPU utilization
        'data_version': 2  # Version 2: Fixed segmentation
    }
    
    # Check for differences
    differences = {}
    for key in current_config:
        if current_config[key] != saved_config.get(key):
            differences[key] = {
                'saved': saved_config.get(key),
                'current': current_config[key]
            }
    
    if differences:
        return False, differences
    return True, None


# --- Data Preparation ---



# --- Main Logic ---

def train():
    files, labels = [], []
    
    # 1. Collect Metadata (Small RAM footprint)
    esc_path = os.path.join(DATASET_PATH, 'ESC-50-master')
    if os.path.exists(esc_path):
        csv = pd.read_csv(os.path.join(esc_path, 'meta', 'esc50.csv'))
        audio_dir = os.path.join(esc_path, 'audio')
        print("Found ESC-50.")
        for _, row in csv.iterrows():
            files.append(os.path.join(audio_dir, row['filename']))
            labels.append(row['category'])

    us8k_path = os.path.join(DATASET_PATH, 'UrbanSound8K')
    if os.path.exists(us8k_path):
        csv = pd.read_csv(os.path.join(us8k_path, 'metadata', 'UrbanSound8K.csv'))
        audio_dir = os.path.join(us8k_path, 'audio')
        print("Found UrbanSound8K.")
        for _, row in csv.iterrows():
            f_path = os.path.join(audio_dir, f"fold{row['fold']}", row['slice_file_name'])
            files.append(f_path)
            labels.append(row['class'])

    if not files:
        print("No datasets found.")
        return

    # 2. Encode Labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)

    # --- Oversampling Wrapper ---
    print("\n--- Balancing Dataset (Oversampling) ---")
    files_series = pd.Series(files)
    labels_series = pd.Series(labels_encoded)
    
    # Calculate counts per class
    counts = labels_series.value_counts()
    max_samples = counts.max()
    print(f"Goal samples per class: {max_samples}")

    balanced_files = []
    balanced_labels = []

    for class_idx in range(num_classes):
        class_files = files_series[labels_series == class_idx].tolist()
        current_count = len(class_files)
        
        # Calculate repeat factor
        # If class has 40 samples and max is 1000, we repeat 25 times
        repeat_factor = max(1, int(round(max_samples / current_count)))
        
        if repeat_factor > 1:
            print(f"Oversampling class '{le.inverse_transform([class_idx])[0]}' x{repeat_factor} ({current_count} -> {current_count * repeat_factor})")
        
        balanced_files.extend(class_files * repeat_factor)
        balanced_labels.extend([class_idx] * (current_count * repeat_factor))

    # 3. Split (on balanced lists)
    print(f"Total balanced samples in list: {len(balanced_files)}")
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(balanced_files, balanced_labels, test_size=0.1, random_state=42)
    
    # 4. TFRecord Generation (Run Once)
    train_tfrecord = os.path.join(DATASET_PATH, 'train.tfrecord')
    val_tfrecord = os.path.join(DATASET_PATH, 'val.tfrecord')
    
    # Check if we need to regenerate (e.g. if file count changed markedly or doesn't exist)
    # For now, simple existence check. 
    # NOTE: If user adds data, they must delete .tfrecord files manually or we implement a force flag.
    
    if not os.path.exists(train_tfrecord):
        print("\n--- Generating Training TFRecords (This happens once) ---")
        save_as_tfrecords(X_train_f, y_train_f, train_tfrecord)
        
    if not os.path.exists(val_tfrecord):
        print("\n--- Generating Validation TFRecords (This happens once) ---")
        save_as_tfrecords(X_test_f, y_test_f, val_tfrecord)
        
    # 5. Load TFRecord Datasets
    print("\n--- Initializing GPU-Accelerated Pipeline ---")
    train_ds = get_tfrecord_dataset(train_tfrecord, augment=True, batch_size=64)
    test_ds = get_tfrecord_dataset(val_tfrecord, augment=False, batch_size=64)
    
    # 5. Model (with checkpoint resume support)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model_epoch_{epoch:02d}.h5')
    latest_checkpoint = None
    initial_epoch = 0
    
    # Check for existing checkpoints
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.h5')])
        if checkpoints:
            latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
            initial_epoch = int(checkpoints[-1].split('_')[-1].split('.')[0])
            
            # Validate training configuration
            config_valid, differences = validate_training_config()
            
            # Check if running interactively (has a terminal) or as a subprocess
            import sys
            is_interactive = sys.stdin.isatty()
            
            if is_interactive:
                # Prompt user for resume or fresh start
                print(f"\n{'='*60}")
                print(f"Found existing checkpoint: {checkpoints[-1]}")
                print(f"Training was at epoch {initial_epoch} out of 30")
                print(f"{'='*60}")
                
                # Warn about configuration changes
                if not config_valid:
                    print(f"\n⚠️  WARNING: Training parameters have changed!")
                    print(f"{'='*60}")
                    for key, values in differences.items():
                        print(f"  {key}: {values['saved']} → {values['current']}")
                    print(f"{'='*60}")
                    print("Resuming with different parameters may produce an inconsistent model.")
                    print("Recommendation: Start fresh training.\n")
                
                while True:
                    response = input("\nDo you want to resume existing model? (yes/no): ").strip().lower()
                    if response in ['yes', 'y']:
                        if not config_valid:
                            confirm = input("Parameters changed. Are you sure? (yes/no): ").strip().lower()
                            if confirm not in ['yes', 'y']:
                                print("Aborting resume. Please restart.")
                                return
                        print(f"Resuming from epoch {initial_epoch}...")
                        break
                    elif response in ['no', 'n']:
                        print("Deleting checkpoint cache and starting from scratch...")
                        # Delete all checkpoints and config
                        for checkpoint_file in checkpoints:
                            os.remove(os.path.join(CHECKPOINT_DIR, checkpoint_file))
                        config_file = os.path.join(CHECKPOINT_DIR, 'training_config.json')
                        if os.path.exists(config_file):
                            os.remove(config_file)
                        latest_checkpoint = None
                        initial_epoch = 0
                        print("Checkpoint cache cleared.")
                        break
                    else:
                        print("Invalid input. Please enter 'yes' or 'no'.")
            else:
                # Non-interactive mode (e.g., called from benchmark_suite.py)
                if not config_valid:
                    print(f"⚠️  WARNING: Training parameters changed! Differences:")
                    for key, values in differences.items():
                        print(f"  {key}: {values['saved']} → {values['current']}")
                    print("Aborting to prevent inconsistent model. Clear checkpoints manually.")
                    return
                # Automatically resume from checkpoint
                print(f"Non-interactive mode detected.")
                print(f"Found checkpoint: {checkpoints[-1]}")
                print(f"Automatically resuming from epoch {initial_epoch}...")
    
    if latest_checkpoint:
        model = tf.keras.models.load_model(latest_checkpoint)
        print(f"Loaded model from {latest_checkpoint}")
    else:
        model = create_cnn_model((N_MELS, FIXED_WIDTH, 1), num_classes)
        # Save config for future validation
        save_training_config()
        print("Starting training from scratch")
    
    # Checkpoint callback - saves after each epoch
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_freq='epoch',
        verbose=1
    )
    
    # Since we use a generator, we need to estimate steps per epoch
    # ESC-50 has 2000 files -> 5 segments each -> augmented 3x = 30k samples
    # US8K has 8700 files -> 4 segments each = 34k samples
    # Total roughly 64k samples / batch 32 = 2000 steps
    print("Starting Training (Parallel Streaming mode)...")
    model.fit(train_ds, 
              epochs=30, 
              initial_epoch=initial_epoch,
              validation_data=test_ds,
              # steps_per_epoch removed to allow dynamic dataset size
              # validation_steps removed
              callbacks=[LogCallback(), checkpoint_callback])
    
    print("Training complete. Starting backup...")
    # Backup before overwriting
    backup_current_model()
    
    print("Backup complete. Saving model...")
    model.save(MODEL_OUTPUT)
    print(f"Model saved to {MODEL_OUTPUT}")
    
    print("Saving label encoder...")
    joblib.dump(le, ENCODER_OUTPUT)
    print(f"Label encoder saved to {ENCODER_OUTPUT}")
    print("All operations complete.")

if __name__ == "__main__":
    train()
