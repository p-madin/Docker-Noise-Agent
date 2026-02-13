import os
import numpy as np
import tensorflow as tf
import librosa
import joblib
from pathlib import Path

# --- Configuration ---
MODEL_PATH = '/root/Workspace/noise_classifier_cnn.h5'
ENCODER_PATH = '/root/Workspace/label_encoder.joblib'
DATASET_PATH = 'datasets'
SAMPLE_RATE = 16000
DURATION = 1.0
N_MELS = 128
FIXED_WIDTH = 128

def extract_mel_spectrogram(audio, sample_rate):
    """Extract mel spectrogram (same as training)."""
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
            
        return mel_db[np.newaxis, ..., np.newaxis]
    except:
        return None

def test_on_random_samples(model, le, num_samples=10):
    """Test classifier on random samples from the dataset."""
    print("\n=== Testing Classifier on Random Dataset Samples ===\n")
    
    # Find all audio files
    audio_files = []
    labels = []
    
    # ESC-50
    esc_path = os.path.join(DATASET_PATH, 'ESC-50-master')
    if os.path.exists(esc_path):
        import pandas as pd
        csv = pd.read_csv(os.path.join(esc_path, 'meta', 'esc50.csv'))
        audio_dir = os.path.join(esc_path, 'audio')
        for _, row in csv.iterrows():
            audio_files.append(os.path.join(audio_dir, row['filename']))
            labels.append(row['category'])
    
    # UrbanSound8K
    us8k_path = os.path.join(DATASET_PATH, 'UrbanSound8K')
    if os.path.exists(us8k_path):
        import pandas as pd
        csv = pd.read_csv(os.path.join(us8k_path, 'metadata', 'UrbanSound8K.csv'))
        audio_dir = os.path.join(us8k_path, 'audio')
        for _, row in csv.iterrows():
            f_path = os.path.join(audio_dir, f"fold{row['fold']}", row['slice_file_name'])
            audio_files.append(f_path)
            labels.append(row['class'])
    
    if not audio_files:
        print("No audio files found in datasets!")
        return
    
    # Random sample
    indices = np.random.choice(len(audio_files), min(num_samples, len(audio_files)), replace=False)
    
    correct = 0
    total = 0
    
    for idx in indices:
        file_path = audio_files[idx]
        true_label = labels[idx]
        
        # Load and process audio
        try:
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            features = extract_mel_spectrogram(audio, sr)
            
            if features is not None:
                # Predict
                probs = model.predict(features, verbose=0)[0]
                pred_idx = np.argmax(probs)
                pred_label = le.inverse_transform([pred_idx])[0]
                confidence = probs[pred_idx]
                
                is_correct = pred_label == true_label
                correct += int(is_correct)
                total += 1
                
                status = "✓" if is_correct else "✗"
                print(f"{status} True: {true_label:20s} | Predicted: {pred_label:20s} | Confidence: {confidence:.2%}")
                
                # Show top 3 predictions
                top3_indices = np.argsort(probs)[-3:][::-1]
                print(f"   Top 3: ", end="")
                for i, tidx in enumerate(top3_indices):
                    print(f"{le.inverse_transform([tidx])[0]} ({probs[tidx]:.1%})", end="")
                    if i < 2:
                        print(", ", end="")
                print("\n")
        except Exception as e:
            print(f"Error processing {file_path}: {e}\n")
    
    if total > 0:
        accuracy = correct / total
        print(f"\n{'='*60}")
        print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")
        print(f"{'='*60}\n")

def test_specific_class(model, le, class_name, num_samples=5):
    """Test classifier on specific class samples."""
    print(f"\n=== Testing '{class_name}' Classification ===\n")
    
    # Find samples of this class
    audio_files = []
    
    # ESC-50
    esc_path = os.path.join(DATASET_PATH, 'ESC-50-master')
    if os.path.exists(esc_path):
        import pandas as pd
        csv = pd.read_csv(os.path.join(esc_path, 'meta', 'esc50.csv'))
        audio_dir = os.path.join(esc_path, 'audio')
        for _, row in csv.iterrows():
            if row['category'] == class_name:
                audio_files.append(os.path.join(audio_dir, row['filename']))
    
    # UrbanSound8K
    us8k_path = os.path.join(DATASET_PATH, 'UrbanSound8K')
    if os.path.exists(us8k_path):
        import pandas as pd
        csv = pd.read_csv(os.path.join(us8k_path, 'metadata', 'UrbanSound8K.csv'))
        audio_dir = os.path.join(us8k_path, 'audio')
        for _, row in csv.iterrows():
            if row['class'] == class_name:
                f_path = os.path.join(audio_dir, f"fold{row['fold']}", row['slice_file_name'])
                audio_files.append(f_path)
    
    if not audio_files:
        print(f"No samples found for class '{class_name}'")
        return
    
    print(f"Found {len(audio_files)} samples. Testing on {min(num_samples, len(audio_files))}...\n")
    
    # Test samples
    indices = np.random.choice(len(audio_files), min(num_samples, len(audio_files)), replace=False)
    
    correct = 0
    confidences = []
    
    for idx in indices:
        file_path = audio_files[idx]
        
        try:
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            features = extract_mel_spectrogram(audio, sr)
            
            if features is not None:
                probs = model.predict(features, verbose=0)[0]
                pred_idx = np.argmax(probs)
                pred_label = le.inverse_transform([pred_idx])[0]
                confidence = probs[pred_idx]
                
                is_correct = pred_label == class_name
                correct += int(is_correct)
                confidences.append(confidence)
                
                status = "✓" if is_correct else "✗"
                print(f"{status} Predicted: {pred_label:20s} | Confidence: {confidence:.2%}")
                
                if not is_correct:
                    # Show what it thought it was
                    print(f"   File: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error: {e}\n")
    
    if confidences:
        avg_conf = np.mean(confidences)
        print(f"\nAccuracy: {correct}/{len(confidences)} = {correct/len(confidences):.2%}")
        print(f"Average Confidence: {avg_conf:.2%}\n")

def main():
    print("Loading model and encoder...")
    model = tf.keras.models.load_model(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    
    print(f"Model loaded. Classes: {len(le.classes_)}")
    print(f"Available classes: {', '.join(le.classes_[:10])}{'...' if len(le.classes_) > 10 else ''}\n")
    
    # Test on random samples
    test_on_random_samples(model, le, num_samples=20)
    
    # Test specific classes that we're trying to dream about
    test_classes = ['siren', 'jackhammer', 'chirping_birds', 'rain', 'mouse_click', 'children_playing']
    
    for cls in test_classes:
        if cls in le.classes_:
            test_specific_class(model, le, cls, num_samples=5)

if __name__ == "__main__":
    main()
