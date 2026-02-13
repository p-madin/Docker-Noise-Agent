import os
import pandas as pd

DATASET_PATH = 'datasets'

# Count ESC-50
esc_path = os.path.join(DATASET_PATH, 'ESC-50-master')
esc_count = 0
esc_classes = 0
if os.path.exists(esc_path):
    csv = pd.read_csv(os.path.join(esc_path, 'meta', 'esc50.csv'))
    esc_count = len(csv)
    esc_classes = csv['category'].nunique()
    print(f"ESC-50: {esc_count} samples, {esc_classes} classes")
    print(f"  Samples per class: {esc_count // esc_classes}")

# Count UrbanSound8K
us8k_path = os.path.join(DATASET_PATH, 'UrbanSound8K')
us8k_count = 0
us8k_classes = 0
if os.path.exists(us8k_path):
    csv = pd.read_csv(os.path.join(us8k_path, 'metadata', 'UrbanSound8K.csv'))
    us8k_count = len(csv)
    us8k_classes = csv['class'].nunique()
    print(f"\nUrbanSound8K: {us8k_count} samples, {us8k_classes} classes")
    
    # Show class distribution
    class_counts = csv['class'].value_counts()
    print(f"  Min samples per class: {class_counts.min()}")
    print(f"  Max samples per class: {class_counts.max()}")
    print(f"  Avg samples per class: {class_counts.mean():.0f}")

# Combined totals
total_samples = esc_count + us8k_count
combined_classes = esc_classes + us8k_classes  # May have overlap
print(f"\n{'='*50}")
print(f"TOTAL: {total_samples} raw samples")
print(f"Combined classes: ~{combined_classes} (may have overlap)")
print(f"\nAfter data segmentation fix (first 1s only):")
print(f"  Base samples: {total_samples}")
print(f"  With augmentation (3x): {total_samples * 3}")
print(f"  Avg per class (with aug): {(total_samples * 3) // combined_classes}")
print(f"{'='*50}")

# Dynamic assessment based on actual counts
avg_per_class = (total_samples * 3) // combined_classes if combined_classes > 0 else 0

print("\nDataset Sufficiency Assessment:")
print(f"  Average samples per class (with augmentation): {avg_per_class}")

# Thresholds (industry standard estimates)
THRESHOLD_BASIC = 100      # Minimum for basic classification
THRESHOLD_GOOD = 500       # Good for most applications
THRESHOLD_PRODUCTION = 1000  # Production quality
THRESHOLD_SOTA = 5000      # State-of-the-art

if avg_per_class >= THRESHOLD_SOTA:
    print(f"  ✓ Excellent: {avg_per_class} >= {THRESHOLD_SOTA} (SOTA quality)")
elif avg_per_class >= THRESHOLD_PRODUCTION:
    print(f"  ✓ Very Good: {avg_per_class} >= {THRESHOLD_PRODUCTION} (Production quality)")
elif avg_per_class >= THRESHOLD_GOOD:
    print(f"  ✓ Good: {avg_per_class} >= {THRESHOLD_GOOD} (Solid classification)")
elif avg_per_class >= THRESHOLD_BASIC:
    print(f"  ⚠ Adequate: {avg_per_class} >= {THRESHOLD_BASIC} (Basic classification)")
else:
    print(f"  ✗ Insufficient: {avg_per_class} < {THRESHOLD_BASIC} (Need more data)")

print(f"\nRecommendation:")
if avg_per_class >= THRESHOLD_GOOD:
    print(f"  Dataset is SUFFICIENT. Focus on model architecture and training.")
elif avg_per_class >= THRESHOLD_BASIC:
    print(f"  Dataset is MARGINAL. Consider adding more data or better augmentation.")
else:
    print(f"  Dataset is TOO SMALL. Collect more data before proceeding.")
