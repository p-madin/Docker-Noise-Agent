import pandas as pd
import os

def analyze_datasets():
    print("=== Dataset Class Inspection ===\n")
    
    # In Docker, app/ is mapped to /root/Workspace
    base_path = '/root/Workspace'
    
    # 1. ESC-50 Analysis
    esc_csv = os.path.join(base_path, 'datasets/ESC-50-master/meta/esc50.csv')
    if os.path.exists(esc_csv):
        df_esc = pd.read_csv(esc_csv)
        print("--- ESC-50 Distribution ---")
        print(f"Total Samples: {len(df_esc)}")
        counts = df_esc['category'].value_counts()
        print(f"Unique Classes: {len(counts)}")
        print(f"Samples per Class: {counts.iloc[0]} (Balanced)\n")
        
        # Look for the specific problem classes
        targets = ['chirping_birds', 'mouse_click', 'rain']
        print(f"Target Classes in ESC-50:")
        for t in targets:
            count = len(df_esc[df_esc['category'] == t])
            print(f"- {t}: {count}")
        print("")
    else:
        print(f"ESC-50 metadata not found at {esc_csv}\n")

    # 2. UrbanSound8K Analysis
    us8k_csv = os.path.join(base_path, 'datasets/UrbanSound8K/metadata/UrbanSound8K.csv')
    if os.path.exists(us8k_csv):
        df_us8 = pd.read_csv(us8k_csv)
        print("--- UrbanSound8K Distribution ---")
        print(f"Total Samples: {len(df_us8)}")
        counts = df_us8['class'].value_counts()
        print(f"Unique Classes: {len(counts)}")
        
        # Look for the specific problem classes
        # US8K classes: [air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music]
        targets_us8k = ['jackhammer', 'drilling', 'dog_bark']
        print(f"\nTarget Classes in US8K:")
        for t in targets_us8k:
            count = counts.get(t, 0)
            print(f"- {t}: {count}")
        
        print("\n--- Discrepancy Analysis ---")
        print(f"1. Extreme Imbalance: Birds/Clicks (ESC-50) only have 40 samples each.")
        print(f"2. Overpowering Classes: UrbanSound8K classes (like Jackhammer) have ~1000 samples each.")
        print(f"3. Confusion Risk: 'drilling' and 'jackhammer' in US8K total {counts.get('drilling', 0) + counts.get('jackhammer', 0)} samples, while 'rain' only has 40 (ESC-50).")
        print(f"4. Missing overlap: US8K doesn't have 'rain' or 'birds', so the model is 25x more likely to pick a US8K class based on volume alone.")
    else:
        print(f"UrbanSound8K metadata not found at {us8k_csv}\n")

if __name__ == "__main__":
    analyze_datasets()
