#!/usr/bin/env python3
"""
Augment eye dataset to ~1000 images using rotation, brightness, flips.
"""

import cv2
import json
import numpy as np
from pathlib import Path
import random

def augment_image(image):
    """Apply random augmentations and return list of augmented images."""
    augmented = []

    # Original
    augmented.append(('orig', image.copy()))

    # Rotation variations (-15 to +15 degrees)
    for angle in [-10, 10]:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append((f'rot{angle}', rotated))

    # Brightness variations
    for beta in [-30, 30]:
        bright = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
        augmented.append((f'bright{beta}', bright))

    # Horizontal flip
    flipped = cv2.flip(image, 1)
    augmented.append(('flip', flipped))

    return augmented

def augment_dataset(input_json, images_dir, output_dir, output_json):
    """Augment all images and create new COCO annotations."""

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    with open(input_json) as f:
        coco_data = json.load(f)

    # Build image_id to annotation mapping
    img_to_ann = {}
    for ann in coco_data['annotations']:
        img_to_ann[ann['image_id']] = ann

    new_images = []
    new_annotations = []
    img_id = 0
    ann_id = 0

    total_original = len(coco_data['images'])
    print(f"Augmenting {total_original} images...")

    for img_info in coco_data['images']:
        img_path = images_dir / img_info['file_name']

        if not img_path.exists():
            continue

        if img_info['id'] not in img_to_ann:
            continue

        ann = img_to_ann[img_info['id']]
        category_id = ann['category_id']

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        augmented_images = augment_image(image)

        for suffix, aug_img in augmented_images:
            img_id += 1
            new_filename = f"{img_path.stem}_{suffix}{img_path.suffix}"

            # Save augmented image
            cv2.imwrite(str(output_dir / new_filename), aug_img)

            h, w = aug_img.shape[:2]
            new_images.append({
                "id": img_id,
                "file_name": new_filename,
                "width": w,
                "height": h
            })

            ann_id += 1
            new_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [0, 0, w, h],
                "area": w * h,
                "iscrowd": 0
            })

    new_coco = {
        "info": {"description": "Augmented eye state dataset"},
        "categories": coco_data['categories'],
        "images": new_images,
        "annotations": new_annotations
    }

    with open(output_json, 'w') as f:
        json.dump(new_coco, f, indent=2)

    # Count per category
    cat_counts = {}
    for ann in new_annotations:
        cat_id = ann['category_id']
        cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1

    print(f"\nDone! Created {len(new_images)} augmented images")
    print(f"  Eyes open: {cat_counts.get(1, 0)}")
    print(f"  Eyes closed: {cat_counts.get(2, 0)}")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    augment_dataset(
        input_json="/home/quinn/eye_detection_project/relabeled_coco.json",
        images_dir="/home/quinn/eye_detection_project/Augen_Bilder",
        output_dir="/home/quinn/eye_detection_project/augmented_images",
        output_json="/home/quinn/eye_detection_project/augmented_coco.json"
    )
