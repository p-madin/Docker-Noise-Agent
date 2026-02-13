import numpy as np
import io
import librosa
import joblib
import tensorflow as tf
from scipy.io import wavfile

class NoiseClassifier:
    def __init__(self):
        self.model_path = 'noise_classifier_cnn.h5'
        self.encoder_path = 'label_encoder.joblib'
        self.model = None
        self.le = None
        
        # Consistent with training
        self.SAMPLE_RATE = 16000
        self.DURATION = 1.0
        self.N_MELS = 128
        self.FIXED_WIDTH = 128
        
        self.load_model()

    def load_model(self):
        """Load the trained CNN model and label encoder."""
        try:
            import os
            if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
                self.model = tf.keras.models.load_model(self.model_path)
                self.le = joblib.load(self.encoder_path)
                print("CNN Model loaded successfully.", flush=True)
            else:
                print(f"Warning: Model not found at {self.model_path}. Please train the model.", flush=True)
        except Exception as e:
            print(f"Error loading model: {e}", flush=True)

    def extract_features(self, audio, sample_rate):
        """Generates a high-resolution Mel Spectrogram with standardized normalization."""
        try:
            # 1. Standardize Amplitude (-1.0 to 1.0 range)
            max_amp = np.max(np.abs(audio))
            if max_amp > 0:
                audio = audio / max_amp

            # 2. Normalize length to 1.0s (padding or cropping)
            target_len = int(sample_rate * self.DURATION)
            if len(audio) > target_len:
                audio = audio[:target_len]
            else:
                audio = np.pad(audio, (0, target_len - len(audio)))

            # 3. Create Mel Spectrogram
            mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=self.N_MELS, n_fft=2048, hop_length=512)
            
            # 4. Standardized DB scaling (fixed ref instead of np.max)
            mel_db = librosa.power_to_db(mel, ref=1.0)
            
            # 5. Ensure fixed width (time frames)
            if mel_db.shape[1] > self.FIXED_WIDTH:
                mel_db = mel_db[:, :self.FIXED_WIDTH]
            else:
                mel_db = np.pad(mel_db, ((0, 0), (0, self.FIXED_WIDTH - mel_db.shape[1])))
                
            # 6. Add channel dimension for CNN (H, W, 1) and batch dimension
            return mel_db[np.newaxis, ..., np.newaxis]
        except Exception as e:
            print(f"Feature extraction error: {e}", flush=True)
            return None

    def classify(self, audio_bytes):
        """
        Classifies raw audio bytes using the CNN model.
        """
        try:
            # 1. Load and Normalize Audio
            samplerate, data = wavfile.read(io.BytesIO(audio_bytes))
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            
            # 2. Heuristic check for silence (RMS energy)
            rms = np.sqrt(np.mean(np.square(data)))
            max_val = np.max(np.abs(data))
            
            if rms < 0.001: 
                return "Silence", 0.0

            # 3. Feature Extraction (2D Spectrogram)
            features = self.extract_features(data, self.SAMPLE_RATE)

            # 4. CNN Inference
            if self.model and self.le and features is not None:
                probs = self.model.predict(features, verbose=0)[0]
                idx = np.argmax(probs)
                confidence = probs[idx]
                label = self.le.inverse_transform([idx])[0]
                
                print(f"Debug: Signal RMS: {rms:.4f}, SampleRate: {samplerate}", flush=True)
                print(f"Debug: CNN detected '{label}' with confidence {confidence:.2f}", flush=True)
                
                # CNN models usually have much higher confidence gradients
                # High precision mode: Only accept very confident predictions
                if confidence >= 0.95: 
                    return label, float(confidence)
                else:
                    print(f"Debug: Low confidence ({confidence:.2f}), reporting as Silence.", flush=True)
            
            return "Silence", 0.0

        except Exception as e:
            print(f"Error in classification: {e}", flush=True)
            return "Error", 0.0

classifier = NoiseClassifier()
