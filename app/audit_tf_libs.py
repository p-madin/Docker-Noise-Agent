import os
import subprocess

tf_path = os.path.dirname(os.path.abspath(__import__('tensorflow').__file__))
target_so = os.path.join(tf_path, 'python', '_pywrap_tensorflow_internal.so')

print(f"Auditing: {target_so}")
if not os.path.exists(target_so):
    print("Error: Library not found.")
else:
    try:
        result = subprocess.run(['ldd', target_so], capture_output=True, text=True)
        print(result.stdout)
        
        print("\nSearching for missing libraries (not found):")
        for line in result.stdout.split('\n'):
            if "not found" in line:
                print(line)
    except Exception as e:
        print(f"Error running ldd: {e}")
