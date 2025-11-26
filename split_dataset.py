import os
import shutil
import random

DATASET_DIR = r"C:\Users\guruprasad\Desktop\BrainTumor\dataset"
CLASSES = ["GLIOMA", "MENINGIOMA", "PITUITARY"]

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

for cls in CLASSES:
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)

    src = os.path.join(DATASET_DIR, cls)
    all_images = os.listdir(src)

    random.shuffle(all_images)

    split = int(0.8 * len(all_images))

    train_imgs = all_images[:split]
    test_imgs = all_images[split:]

    for img in train_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(TRAIN_DIR, cls))

    for img in test_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(TEST_DIR, cls))

print("Dataset successfully split into train/test folders!")
