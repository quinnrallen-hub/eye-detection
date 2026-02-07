#!/usr/bin/env python3
"""
Crop eye regions from full-frame images using MediaPipe.
Creates a new dataset of cropped eye images for training.
"""

import cv2
import json
import numpy as np
from pathlib import Path
import mediapipe as mp

# Eye landmark indices
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249,
                    263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155,
                     133, 173, 157, 158, 159, 160, 161, 246]

PADDING = 0.5  # Extra padding around eye
OUTPUT_DIR = Path("/home/quinn/eye_detection_project/cropped_eyes")
OUTPUT_JSON = "/home/quinn/eye_detection_project/cropped_eyes_coco.json"


def crop_eye(frame, landmarks, indices):
    """Crop eye region with padding."""
    h, w = frame.shape[:2]

    xs = [landmarks[i].x * w for i in indices]
    ys = [landmarks[i].y * h for i in indices]

    cx, cy = np.mean(xs), np.mean(ys)
    size = max(max(xs) - min(xs), max(ys) - min(ys))
    half = size * (0.5 + PADDING)

    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(w, cx + half))
    y2 = int(min(h, cy + half))

    if x2 - x1 < 20 or y2 - y1 < 20:
        return None

    return frame[y1:y2, x1:x2].copy()


def main():
    # Load original annotations
    with open("/home/quinn/eye_detection_project/augmented_coco.json") as f:
        data = json.load(f)

    img_dir = Path("/home/quinn/eye_detection_project/augmented_images")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Setup MediaPipe
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path="/home/quinn/eye_detection_project/face_landmarker.task"
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    id_to_file = {img['id']: img['file_name'] for img in data['images']}
    id_to_cat = {ann['image_id']: ann['category_id'] for ann in data['annotations']}

    new_images = []
    new_annotations = []
    img_id = 0
    ann_id = 0

    skipped = 0

    for orig_img in data['images']:
        img_file = orig_img['file_name']
        img_path = img_dir / img_file
        category_id = id_to_cat.get(orig_img['id'])

        if not img_path.exists() or category_id is None:
            skipped += 1
            continue

        # Read and detect
        frame = cv2.imread(str(img_path))
        if frame is None:
            skipped += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        if not result.face_landmarks:
            skipped += 1
            continue

        lm = result.face_landmarks[0]

        # Crop both eyes
        for eye_name, indices in [("left", LEFT_EYE_CONTOUR), ("right", RIGHT_EYE_CONTOUR)]:
            crop = crop_eye(frame, lm, indices)
            if crop is None:
                continue

            # Save cropped image
            base_name = img_path.stem
            crop_name = f"{base_name}_{eye_name}.jpg"
            crop_path = OUTPUT_DIR / crop_name
            cv2.imwrite(str(crop_path), crop)

            img_id += 1
            ann_id += 1

            h, w = crop.shape[:2]
            new_images.append({
                "id": img_id,
                "file_name": crop_name,
                "width": w,
                "height": h
            })
            new_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [0, 0, w, h],
                "area": w * h,
                "iscrowd": 0
            })

    landmarker.close()

    # Save new COCO JSON
    new_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": data['categories']
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(new_data, f, indent=2)

    print(f"Created {len(new_images)} cropped eye images")
    print(f"Skipped {skipped} images (no face detected)")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Annotations: {OUTPUT_JSON}")

    # Count by category
    open_count = sum(1 for a in new_annotations if a['category_id'] == 1)
    closed_count = sum(1 for a in new_annotations if a['category_id'] == 2)
    print(f"  Eyes open: {open_count}")
    print(f"  Eyes closed: {closed_count}")


if __name__ == "__main__":
    main()
