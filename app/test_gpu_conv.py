import tensorflow as tf
import os

print("--- GPU Test Script ---")
print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {gpus}")

if not gpus:
    print("Error: No GPU detected by TensorFlow.")
    exit(1)

try:
    with tf.device('/GPU:0'):
        print("Attempting a 2D convolution on GPU...")
        a = tf.random.normal([1, 64, 64, 3])
        conv = tf.keras.layers.Conv2D(16, 3, activation='relu')
        b = conv(a)
        print(f"Success! Output shape: {b.shape}")
except Exception as e:
    print(f"Failure during GPU operation: {e}")
    # Print environment for debugging
    print("\nEnvironment Variables:")
    for k, v in os.environ.items():
        if "CUDA" in k or "TF" in k or "LD" in k:
            print(f"{k}={v}")
