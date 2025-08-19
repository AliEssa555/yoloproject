import os
import cv2
import albumentations as A
import shutil
from pathlib import Path

# Path configs
ORIG_IMG_DIR = "data/train/images"
ORIG_LABEL_DIR = "data/train/labels"
AUG_IMG_DIR = "augmented_data/train/images"
AUG_LABEL_DIR = "augmented_data/train/labels"

# Create output dirs
Path(AUG_IMG_DIR).mkdir(parents=True, exist_ok=True)
Path(AUG_LABEL_DIR).mkdir(parents=True, exist_ok=True)

# Albumentations augmentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=25, p=0.7),
    A.Affine(translate_percent=0.1, shear=15, p=0.7),
    A.Mosaic(p=0.3),
    A.CoarseDropout(num_holes_range=(3, 6),hole_height_range=(10, 20),hole_width_range=(10, 20),fill="random_uniform",p=1.0)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Load images and apply transforms
for filename in os.listdir(ORIG_IMG_DIR):
    if not filename.endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(ORIG_IMG_DIR, filename)
    label_path = os.path.join(ORIG_LABEL_DIR, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        bboxes = []
        class_labels = []
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x_center, y_center, w, h = map(float, parts)
            bboxes.append([x_center, y_center, w, h])
            class_labels.append(int(cls))

    # Apply transform
    try:
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    except:
        continue  # skip if bbox transform fails

    aug_img = augmented["image"]
    aug_bboxes = augmented["bboxes"]
    aug_classes = augmented["class_labels"]

    # Save augmented image
    out_img_path = os.path.join(AUG_IMG_DIR, filename)
    cv2.imwrite(out_img_path, aug_img)

    # Save label
    out_label_path = os.path.join(AUG_LABEL_DIR, filename.replace(".jpg", ".txt").replace(".png", ".txt"))
    with open(out_label_path, "w") as f:
        for cls, bbox in zip(aug_classes, aug_bboxes):
            f.write(f"{cls} {' '.join([str(round(x, 6)) for x in bbox])}\n")
