import os
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_PATH = '/root/Workspace/noise_classifier_cnn.h5'
ENCODER_PATH = '/root/Workspace/label_encoder.joblib'
OUTPUT_DIR = '/root/Workspace/dreams'
ITERATIONS = 300
LEARNING_RATE = 2.0  # Larger jumps for dB scale

def generate_dream(model, le, target_class_idx):
    class_name = le.classes_[target_class_idx]
    print(f"\n--- Dreaming of '{class_name}' (Index: {target_class_idx}) ---")
    
    # Octave parameters
    OCTAVE_SCALE = 1.4
    OCTAVES = 3
    
    # 1. Initialize with Pink Noise
    rows, cols = 128, 128
    pink_noise = np.random.randn(rows, cols)
    f_noise = np.fft.fft2(pink_noise)
    fx, fy = np.meshgrid(np.fft.fftfreq(cols), np.fft.fftfreq(rows))
    f_radius = np.sqrt(fx**2 + fy**2)
    f_radius[0,0] = 1.0
    pink_noise = np.real(np.fft.ifft2(f_noise / f_radius))
    pink_noise = (pink_noise - pink_noise.mean()) / pink_noise.std()
    base_img = (pink_noise * 10.0) - 50.0

    img = base_img.astype('float32') # Start with full res base
    
    # Prepare octaves (scales)
    # We will start small and upscale
    original_shape = img.shape
    shapes = [original_shape]
    for _ in range(OCTAVES - 1):
        shapes.append((int(shapes[-1][0] / OCTAVE_SCALE), int(shapes[-1][1] / OCTAVE_SCALE)))
    shapes = shapes[::-1] # Smallest to largest

    input_img = tf.Variable(np.zeros(shapes[0]) [np.newaxis, ..., np.newaxis], dtype=tf.float32)

    # Gaussian kernel for smoothing
    def get_gaussian_kernel(size=3, sigma=1.0):
        x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
        g = tf.exp(-(x**2) / (2 * sigma**2))
        g = g / tf.reduce_sum(g)
        return g[:, tf.newaxis] * g[tf.newaxis, :]
    gaussian_kernel = get_gaussian_kernel(size=3, sigma=0.5)[..., tf.newaxis, tf.newaxis]

    for octave_idx, shape in enumerate(shapes):
        print(f"  Processing Octave {octave_idx+1}/{OCTAVES} (Shape: {shape})")
        
        # Upscale previous result to new shape
        if octave_idx > 0:
            # Upscale current image
            img_tensor = tf.image.resize(input_img, shape)
            
            # Blend with original noise at this scale to add detail
            # (Optional: can just use upscaled version)
            input_img = tf.Variable(img_tensor)
        else:
             # First octave: simple resize of base noise
             img_tensor = tf.image.resize(base_img[np.newaxis, ..., np.newaxis], shape)
             input_img = tf.Variable(img_tensor)

        # Optimization loop for this octave
        for i in range(ITERATIONS):
            with tf.GradientTape() as tape:
                tape.watch(input_img)
                # Resize to model input size (128x128) for prediction
                # The model *expects* 128x128. We optimize at lower res, then resize up to feed model.
                model_input = tf.image.resize(input_img, (128, 128))
                
                preds = model(model_input, training=False)
                loss = -preds[0, target_class_idx]

            grads = tape.gradient(loss, input_img)
            
            # Smooth gradients
            grads = tf.nn.conv2d(grads, gaussian_kernel, strides=[1,1,1,1], padding='SAME')
            grads /= (tf.math.reduce_std(grads) + 1e-8)
            
            input_img.assign_sub(grads * LEARNING_RATE)
            input_img.assign(tf.clip_by_value(input_img, -80.0, 0.0))
            
            if i % 50 == 0:
                 print(f"    Iter {i}: Loss {loss.numpy():.4f}")

    # 3. Post-processing
    # Final resize to ensuring exact 128x128
    final_img = tf.image.resize(input_img, (128, 128))
    final_spec = final_img.numpy()[0, :, :, 0]
    
    print(f"  Final activation: {preds[0, target_class_idx].numpy():.6f}")
    print(f"  Spectrogram range: [{final_spec.min():.2f}, {final_spec.max():.2f}] dB")
    
    # 4. Audio Reconstruction (Griffin-Lim)
    print("  Reconstructing audio...")
    # Map back from dB to amplitude
    S_db = final_spec
    S_amp = librosa.db_to_amplitude(S_db)
    
    print(f"  Amplitude range: [{S_amp.min():.6f}, {S_amp.max():.6f}]")
    
    # Use more iterations for better quality
    audio = librosa.feature.inverse.mel_to_audio(
        S_amp, 
        sr=16000, 
        n_fft=2048, 
        hop_length=512, 
        n_iter=128  # Increased from 64 for better quality
    )
    
    # Normalize audio to prevent clipping
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    audio = audio * 0.95  # Scale to 95% to avoid clipping
    
    print(f"  Audio duration: {len(audio)/16000:.2f}s")
    print(f"  Audio range: [{audio.min():.6f}, {audio.max():.6f}]")
    
    # 5. Save Output
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    filename = os.path.join(OUTPUT_DIR, f"dream_{class_name}.wav")
    sf.write(filename, audio, 16000)
    print(f"  Saved to: {filename}")
    
    plt.figure(figsize=(10, 4))
    plt.imshow(final_spec, aspect='auto', origin='lower', cmap='viridis')
    plt.title(f"AI Dream: {class_name}")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Frequency Bins")
    plt.colorbar(label='dB')
    plt.savefig(os.path.join(OUTPUT_DIR, f"dream_{class_name}.png"))
    plt.close()

def main():
    import joblib
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print("Error: Model or Encoder not found.")
        return

    print("Loading model and encoder...")
    model = tf.keras.models.load_model(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    
    # Dream of the core noise classes
    targets = ['siren', 'jackhammer', 'chirping_birds', 'rain', 'mouse_click', 'children_playing']
    
    for t in targets:
        if t in le.classes_:
            idx = list(le.classes_).index(t)
            generate_dream(model, le, idx)
        else:
            print(f"Class '{t}' not found in encoder.")

if __name__ == "__main__":
    main()
