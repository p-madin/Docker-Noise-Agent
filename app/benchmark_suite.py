import time
import os
import subprocess
import sys

def run_training_and_measure():
    print("=== AI-Noise Training Benchmark ===")
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Clear any existing checkpoints to ensure a full training benchmark
    checkpoint_dir = "/root/Workspace/checkpoints"
    if os.path.exists(checkpoint_dir):
        import shutil
        print(f"Clearing existing checkpoints for consistent benchmark...")
        shutil.rmtree(checkpoint_dir)
        print(f"Checkpoint cache cleared. Starting fresh training run.")
    
    start_total = time.time()
    
    # We will wrap the existing compile_classifiers.py logic 
    # and use some internal timing hooks if possible, or just time the subprocess.
    # For a high-fidelity benchmark, we'll run it as a subprocess to capture overhead.
    
    # Run directly inside the container
    cmd = ["python", "/root/Workspace/compile_classifiers.py"]
    
    # Re-disabling cuDNN autotune as it causes hangs/delays on this GPU
    env = os.environ.copy()
    env["TF_CUDNN_USE_AUTOTUNE"] = "0"
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    
    file_count = 0
    epoch_start_time = None
    epoch_times = []
    
    # Using 'for line in process.stdout' is the standard way to read until EOF.
    # It will not loop infinitely unless the subprocess produces infinite output.
    for line in process.stdout:
        print(line.strip())
        
        # Parse output for metrics (specific to our compile_classifiers.py logs)
        if "Found ESC-50" in line or "Found UrbanSound8K" in line:
            continue
            
        if "Initializing parallel data pipeline" in line or "Initializing GPU-Accelerated Pipeline" in line:
            load_start = time.time()
            
        if "Starting Training" in line:
            load_end = time.time()
            print(f"\n[BENCHMARK] Data Loading Phase: {load_end - load_start:.2f}s")
            
        if "Epoch" in line and "/" in line and ":" in line:
            # Detect start of a new epoch
            epoch_start_time = time.time()
            
        if "accuracy:" in line and "val_accuracy:" in line:
            # Detect end of an epoch
            if epoch_start_time:
                duration = time.time() - epoch_start_time
                epoch_times.append(duration)
                print(f"[BENCHMARK] Epoch Duration: {duration:.2f}s")

    process.wait()
    end_total = time.time()
    
    avg_epoch = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    
    print("\n" + "="*40)
    print("FINAL BENCHMARK RESULTS")
    print("="*40)
    print(f"Total Duration:   {end_total - start_total:.2f}s")
    if epoch_times:
        print(f"Avg Epoch Time:   {avg_epoch:.2f}s")
    print(f"Accuracy Reached: (See logs above)")
    print("="*40)

if __name__ == "__main__":
    try:
        run_training_and_measure()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
