#!/usr/bin/env python3
"""
Relabel eye images using MediaPipe FaceLandmarker.
Detects if eyes are open or closed using Eye Aspect Ratio (EAR).
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import os
from pathlib import Path
import numpy as np

# Eye landmark indices for MediaPipe Face Landmarker
# Left eye vertical points
LEFT_EYE_TOP = [159, 158, 157, 173]
LEFT_EYE_BOTTOM = [145, 144, 153, 154]
LEFT_EYE_LEFT = [33]
LEFT_EYE_RIGHT = [133]

# Right eye vertical points
RIGHT_EYE_TOP = [386, 385, 384, 398]
RIGHT_EYE_BOTTOM = [374, 373, 380, 381]
RIGHT_EYE_LEFT = [362]
RIGHT_EYE_RIGHT = [263]

def calculate_ear(landmarks, eye_top, eye_bottom, eye_left, eye_right):
    """Calculate Eye Aspect Ratio (EAR)"""
    def get_point(idx):
        return np.array([landmarks[idx].x, landmarks[idx].y])

    # Vertical distances
    v1 = np.linalg.norm(get_point(eye_top[0]) - get_point(eye_bottom[0]))
    v2 = np.linalg.norm(get_point(eye_top[-1]) - get_point(eye_bottom[-1]))

    # Horizontal distance
    h = np.linalg.norm(get_point(eye_left[0]) - get_point(eye_right[0]))

    if h == 0:
        return 0

    ear = (v1 + v2) / (2.0 * h)
    return ear

def create_landmarker():
    """Create MediaPipe FaceLandmarker"""
    base_options = python.BaseOptions(
        model_asset_path=str(Path(__file__).parent / 'face_landmarker.task')
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

def detect_eye_state(landmarker, image_path, ear_threshold=0.2):
    """
    Detect if eyes are open or closed in an image.
    Returns: 'open', 'closed', or None if no face detected
    """
    image = mp.Image.create_from_file(str(image_path))

    results = landmarker.detect(image)

    if not results.face_landmarks:
        return None, 0, None

    landmarks = results.face_landmarks[0]

    # Calculate EAR for both eyes
    left_ear = calculate_ear(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                              LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
    right_ear = calculate_ear(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                               RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)

    avg_ear = (left_ear + right_ear) / 2.0

    # Also check blendshapes for eye closure if available
    blendshapes = None
    if results.face_blendshapes:
        blendshapes = {bs.category_name: bs.score for bs in results.face_blendshapes[0]}
        # eyeBlinkLeft and eyeBlinkRight indicate closure
        left_blink = blendshapes.get('eyeBlinkLeft', 0)
        right_blink = blendshapes.get('eyeBlinkRight', 0)
        avg_blink = (left_blink + right_blink) / 2.0

        # Use blendshape blink score (higher = more closed)
        if avg_blink > 0.5:
            return 'closed', avg_ear, blendshapes
        elif avg_blink < 0.3:
            return 'open', avg_ear, blendshapes

    # Fall back to EAR threshold
    if avg_ear < ear_threshold:
        return 'closed', avg_ear, blendshapes
    else:
        return 'open', avg_ear, blendshapes

def relabel_dataset(images_dir, output_json, ear_threshold=0.2):
    """
    Process all images and create new COCO annotations.
    """
    images_dir = Path(images_dir)

    categories = [
        {"id": 1, "name": "eyes_open"},
        {"id": 2, "name": "eyes_closed"}
    ]

    images = []
    annotations = []

    image_files = sorted([f for f in images_dir.iterdir()
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    print(f"Processing {len(image_files)} images...")

    stats = {'open': 0, 'closed': 0, 'no_face': 0}

    landmarker = create_landmarker()

    for idx, img_path in enumerate(image_files):
        try:
            state, ear, blendshapes = detect_eye_state(landmarker, img_path, ear_threshold)
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
            stats['no_face'] += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        images.append({
            "id": idx + 1,
            "file_name": img_path.name,
            "width": w,
            "height": h
        })

        if state == 'open':
            category_id = 1
            stats['open'] += 1
        elif state == 'closed':
            category_id = 2
            stats['closed'] += 1
        else:
            stats['no_face'] += 1
            print(f"  No face detected: {img_path.name}")
            continue

        annotations.append({
            "id": len(annotations) + 1,
            "image_id": idx + 1,
            "category_id": category_id,
            "ear_value": round(ear, 4),
            "bbox": [0, 0, w, h],
            "area": w * h,
            "iscrowd": 0
        })

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(image_files)} images...")

    landmarker.close()

    coco_data = {
        "info": {"description": "Eye state detection dataset - relabeled with MediaPipe"},
        "categories": categories,
        "images": images,
        "annotations": annotations
    }

    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"\nDone! Results saved to {output_json}")
    print(f"Stats: {stats}")
    return stats

if __name__ == "__main__":
    _base = Path(__file__).parent
    images_dir = str(_base / "Augen_Bilder")
    output_json = str(_base / "relabeled_coco.json")

    stats = relabel_dataset(images_dir, output_json, ear_threshold=0.2)
