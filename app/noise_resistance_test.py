import os
import numpy as np
import tensorflow as tf
import librosa
import joblib
import pandas as pd
from validate_classifier import extract_mel_spectrogram

# --- Configuration ---
MODEL_PATH = '/root/Workspace/noise_classifier_cnn.h5'
ENCODER_PATH = '/root/Workspace/label_encoder.joblib'
DATASET_PATH = 'datasets'
SAMPLE_RATE = 16000

def add_noise(audio, snr_db):
    """
    Adds Gaussian noise to the audio signal to achieve a specific Signal-to-Noise Ratio (SNR) in dB.
    Lower SNR = More Noise.
    """
    # Calculate signal power
    signal_power = np.mean(audio ** 2)
    
    # Calculate required noise power
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Generate noise
    noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
    
    # Add noise
    noisy_audio = audio + noise
    
    # Clip to valid audio range [-1, 1] if originally normalized, mostly safe for float32
    return noisy_audio.astype(np.float32)

def test_noise_resilience(model, le, num_samples=100):
    print(f"\n=== Testing Noise Resilience (Samples: {num_samples}) ===\n")
    
    # 1. Collect Random Samples
    audio_files = []
    labels = []
    
    # Scan datasets (reuse logic)
    for dataset in ['ESC-50-master', 'UrbanSound8K']:
        path = os.path.join(DATASET_PATH, dataset)
        if os.path.exists(path):
            # Simplistic walk for robustness across both dataset structures
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.wav'):
                        # Try to infer label from known structures or metadata would be better
                        # For speed/simplicity, we load metadata again
                        pass 
    
    # Better approach: Read Metadata for Ground Truth
    # ESC-50
    esc_path = os.path.join(DATASET_PATH, 'ESC-50-master')
    if os.path.exists(esc_path):
        csv = pd.read_csv(os.path.join(esc_path, 'meta', 'esc50.csv'))
        audio_dir = os.path.join(esc_path, 'audio')
        for _, row in csv.iterrows():
            audio_files.append(os.path.join(audio_dir, row['filename']))
            labels.append(row['category'])
            
    # Subsample
    if not audio_files:
        print("No audio files found.")
        return

    indices = np.random.choice(len(audio_files), min(num_samples, len(audio_files)), replace=False)
    test_files = [audio_files[i] for i in indices]
    test_labels = [labels[i] for i in indices]
    
    # 2. Define Noise Levels (SNR dB)
    # 30dB = Very Clear, 20dB = Clear, 10dB = Noisy, 0dB = Very Noisy (Signal = Noise), -5dB = Extreme
    noise_levels = [None, 30, 20, 10, 5, 0] 
    
    results = {}

    for snr in noise_levels:
        correct = 0
        total = 0
        
        print(f"Testing SNR: {snr if snr is not None else 'Clean (Original)'} dB...")
        
        for file_path, true_label in zip(test_files, test_labels):
            try:
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Apply Noise
                if snr is not None:
                    audio = add_noise(audio, snr)
                
                # Extract Features
                features = extract_mel_spectrogram(audio, sr)
                
                if features is not None:
                    probs = model.predict(features, verbose=0)[0]
                    pred_label = le.inverse_transform([np.argmax(probs)])[0]
                    
                    if pred_label == true_label:
                        correct += 1
                    total += 1
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0
        results[str(snr)] = accuracy
        print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})\n")

    # 3. Summary Report
    print("\n=== Resilience Report ===")
    print("SNR (dB) | Accuracy")
    print("---------|----------")
    for snr in noise_levels:
        key = str(snr)
        label = "Clean" if snr is None else f"{snr} dB"
        print(f"{label:8s} | {results[key]:.2%}")

if __name__ == "__main__":
    # Load resources
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    
    test_noise_resilience(model, le)
