import joblib
import os

def list_categories():
    encoder_path = 'label_encoder.joblib'
    
    if not os.path.exists(encoder_path):
        print(f"Error: Label encoder not found at {encoder_path}")
        print("Have you run the training script yet?")
        return

    try:
        le = joblib.load(encoder_path)
        print("\n=== Trained Noise Categories ===")
        classes = sorted(le.classes_)
        for i, category in enumerate(classes, 1):
            print(f"{i:2d}. {category}")
        print(f"================================\nTotal: {len(classes)} categories")
    except Exception as e:
        print(f"Error loading categories: {e}")

if __name__ == "__main__":
    list_categories()
