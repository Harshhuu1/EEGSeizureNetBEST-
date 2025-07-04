import os
import shutil
import random

# Current dataset path (no "train" folder)
base_dir = r"C:\Users\ASUS\PycharmProjects\Langchainmodels\chb01_fast_download\dataset"
val_dir = os.path.join(base_dir, "val")
split_ratio = 0.2  # 20% for validation

classes = ["interictal", "preictal"]

for cls in classes:
    cls_path = os.path.join(base_dir, cls)
    cls_val_path = os.path.join(val_dir, cls)
    os.makedirs(cls_val_path, exist_ok=True)

    files = os.listdir(cls_path)
    files = [f for f in files if f.endswith(".png")]
    random.shuffle(files)

    val_count = int(len(files) * split_ratio)
    val_files = files[:val_count]

    for file in val_files:
        src = os.path.join(cls_path, file)
        dst = os.path.join(cls_val_path, file)
        shutil.move(src, dst)

print("✅ Split complete. 80% remains in dataset/, 20% moved to dataset/val/")
