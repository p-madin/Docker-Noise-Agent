import tensorflow as tf
import os

print("--- GPU Test Script (No Catch) ---")
print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {gpus}")

if not gpus:
    print("Error: No GPU detected by TensorFlow.")
    exit(1)

with tf.device('/GPU:0'):
    print("Attempting a 2D convolution on GPU...")
    # Small tensor to avoid memory issues
    a = tf.random.normal([1, 32, 32, 3])
    conv = tf.keras.layers.Conv2D(8, 3, activation='relu')
    b = conv(a)
    print(f"Success! Output shape: {b.shape}")
