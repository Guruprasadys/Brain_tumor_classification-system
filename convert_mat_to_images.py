import os
import h5py
import numpy as np
import cv2

SOURCE_FOLDERS = [
    r"C:\Users\guruprasad\Downloads\1512427\brainTumorDataPublic_1-766",
    r"C:\Users\guruprasad\Downloads\1512427\brainTumorDataPublic_767-1532",
    r"C:\Users\guruprasad\Downloads\1512427\brainTumorDataPublic_1533-2298",
    r"C:\Users\guruprasad\Downloads\1512427\brainTumorDataPublic_2299-3064",
]

OUTPUT_DIR = r"C:\Users\guruprasad\Desktop\BrainTumor\dataset"

LABELS = {
    1: "MENINGIOMA",
    2: "GLIOMA",
    3: "PITUITARY",
    4: "NOTUMOR"
}

for label in LABELS.values():
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)


def convert_all():
    for folder in SOURCE_FOLDERS:
        print(f"Processing folder: {folder}")

        for file in os.listdir(folder):
            if file.endswith(".mat"):
                file_path = os.path.join(folder, file)

                try:
                    with h5py.File(file_path, "r") as f:
                        # Load image
                        img = np.array(f["cjdata"]["image"])

                        # ---- FIX for CV_16S error ----
                        img = img.astype(np.float32)
                        img = img - img.min()
                        if img.max() != 0:
                            img = img / img.max()
                        img = (img * 255).astype(np.uint8)
                        # --------------------------------

                        # Resize
                        img_resized = cv2.resize(img, (150, 150))

                        # Convert to BGR
                        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

                        # Load label
                        label = int(np.array(f["cjdata"]["label"]))

                        class_name = LABELS.get(label, "UNKNOWN")

                        output_path = os.path.join(
                            OUTPUT_DIR,
                            class_name,
                            file.replace(".mat", ".jpg")
                        )

                        cv2.imwrite(output_path, img_resized)

                except Exception as e:
                    print("Error processing:", file_path)
                    print("Reason:", e)


if __name__ == "__main__":
    print("Starting conversion...")
    convert_all()
    print("Finished! Check dataset folders!")
