#!/usr/bin/env python3
"""
Eye open/closed detection GUI.
Uses OpenCV Haar cascades for eye detection, custom CNN to classify open/closed.
NO MediaPipe - only OpenCV + your trained model.
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
import torch.nn as nn

# --- Config ---
IMG_SIZE = 64
MODEL_PATH = "/home/quinn/eye_detection_project/eye_classifier.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# OpenCV Haar cascades (built into OpenCV)
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


class EyeCNN(nn.Module):
    """CNN with BatchNorm - must match training architecture."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class EyeDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Detector (No MediaPipe)")
        self.root.geometry("800x650")

        # State
        self.cap = None
        self.running = False
        self.threshold = 0.5
        self.frame_count = 0
        self.last_left_prob = None
        self.last_right_prob = None
        self.inference_interval = 3

        # Load model
        self.model = self._load_model()

        # Build UI
        self._setup_ui()

        # Start
        self._start_stream()

    def _load_model(self):
        model = EyeCNN().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.eval()
        return model

    def _setup_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Camera selection
        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=5)

        ttk.Label(top, text="Camera:").pack(side=tk.LEFT, padx=5)
        self.cam_var = tk.IntVar(value=0)
        ttk.Spinbox(top, from_=0, to=4, textvariable=self.cam_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Switch", command=self._switch_camera).pack(side=tk.LEFT, padx=5)

        # Video display
        self.video_canvas = tk.Canvas(main, width=640, height=400, bg="black")
        self.video_canvas.pack(pady=10)

        # Eye crops
        eye_frame = ttk.Frame(main)
        eye_frame.pack(pady=5)

        ttk.Label(eye_frame, text="Left Eye:").pack(side=tk.LEFT, padx=5)
        self.left_canvas = tk.Canvas(eye_frame, width=80, height=60, bg="gray")
        self.left_canvas.pack(side=tk.LEFT, padx=5)

        ttk.Label(eye_frame, text="Right Eye:").pack(side=tk.LEFT, padx=20)
        self.right_canvas = tk.Canvas(eye_frame, width=80, height=60, bg="gray")
        self.right_canvas.pack(side=tk.LEFT, padx=5)

        # Results
        result_frame = ttk.Frame(main)
        result_frame.pack(pady=10)

        self.left_var = tk.StringVar(value="--")
        self.left_label = ttk.Label(result_frame, textvariable=self.left_var, font=("Helvetica", 16, "bold"))
        self.left_label.pack(side=tk.LEFT, padx=30)

        self.status_var = tk.StringVar(value="Starting...")
        self.status_label = ttk.Label(result_frame, textvariable=self.status_var, font=("Helvetica", 22, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=30)

        self.right_var = tk.StringVar(value="--")
        self.right_label = ttk.Label(result_frame, textvariable=self.right_var, font=("Helvetica", 16, "bold"))
        self.right_label.pack(side=tk.LEFT, padx=30)

        # Threshold slider
        thresh_frame = ttk.Frame(main)
        thresh_frame.pack(pady=10, fill=tk.X, padx=50)

        ttk.Label(thresh_frame, text="Threshold:").pack(side=tk.LEFT, padx=5)
        self.thresh_var = tk.DoubleVar(value=0.5)
        self.thresh_slider = ttk.Scale(
            thresh_frame, from_=0.0, to=1.0, variable=self.thresh_var,
            orient=tk.HORIZONTAL, command=self._on_threshold_change
        )
        self.thresh_slider.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.thresh_label = ttk.Label(thresh_frame, text="0.50", width=5)
        self.thresh_label.pack(side=tk.LEFT, padx=5)

    def _switch_camera(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.cam_var.get())
        if not self.running:
            self.running = True
            self._update()

    def _start_stream(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.running = True
            self._update()
        else:
            self.status_var.set("No camera found")

    def _on_threshold_change(self, value):
        self.threshold = float(value)
        self.thresh_label.config(text=f"{self.threshold:.2f}")

    def _preprocess_crop(self, crop):
        """Preprocess a single crop for CNN."""
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return None
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def _predict_batch(self, left_crop, right_crop):
        """Predict both eyes in a single batch for speed."""
        left_tensor = self._preprocess_crop(left_crop)
        right_tensor = self._preprocess_crop(right_crop)

        if left_tensor is None and right_tensor is None:
            return None, None

        try:
            tensors = []
            left_idx, right_idx = -1, -1
            if left_tensor is not None:
                left_idx = len(tensors)
                tensors.append(left_tensor)
            if right_tensor is not None:
                right_idx = len(tensors)
                tensors.append(right_tensor)

            batch = torch.stack(tensors).to(DEVICE)

            with torch.no_grad():
                output = self.model(batch)
                probs = torch.softmax(output, dim=1)

            left_prob = probs[left_idx, 0].item() if left_idx >= 0 else None
            right_prob = probs[right_idx, 0].item() if right_idx >= 0 else None

            return left_prob, right_prob
        except Exception as e:
            print(f"CNN predict error: {e}")
            return None, None

    def _detect_eyes(self, frame):
        """Detect face and estimate eye positions. Returns list of eye boxes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return [], None

        # Use largest face
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

        # Estimate eye positions based on face proportions
        # Eyes are about 40% down from top of face, smaller boxes
        eye_h = int(fh * 0.12)  # Eye height (smaller)
        eye_w = int(fw * 0.20)  # Eye width (smaller)
        eye_y = fy + int(fh * 0.35)  # Eyes at 35% down from top

        # Right eye (left side of image) - note: from camera's perspective
        right_eye_x = fx + int(fw * 0.18)
        # Left eye (right side of image)
        left_eye_x = fx + int(fw * 0.62)

        # Create eye boxes with small padding
        pad = int(eye_w * 0.2)
        right_box = (
            max(0, right_eye_x - pad),
            max(0, eye_y - pad),
            min(frame.shape[1], right_eye_x + eye_w + pad),
            min(frame.shape[0], eye_y + eye_h + pad)
        )
        left_box = (
            max(0, left_eye_x - pad),
            max(0, eye_y - pad),
            min(frame.shape[1], left_eye_x + eye_w + pad),
            min(frame.shape[0], eye_y + eye_h + pad)
        )

        face_box = (fx, fy, fx + fw, fy + fh)
        return [right_box, left_box], face_box

    def _update(self):
        if not self.running:
            return

        if not self.cap or not self.cap.isOpened():
            self.root.after(30, self._update)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self._update)
            return

        h, w = frame.shape[:2]
        display = frame.copy()

        # Detect eyes
        eye_boxes, face_box = self._detect_eyes(frame)

        left_prob, right_prob = None, None
        left_crop, right_crop = None, None

        if len(eye_boxes) == 2:
            # Eyes detected - right eye is first (left side of image)
            right_box = eye_boxes[0]
            left_box = eye_boxes[1]

            right_crop = frame[right_box[1]:right_box[3], right_box[0]:right_box[2]]
            left_crop = frame[left_box[1]:left_box[3], left_box[0]:left_box[2]]

            # Run CNN every N frames
            self.frame_count += 1
            if self.frame_count >= self.inference_interval:
                self.frame_count = 0
                self.last_left_prob, self.last_right_prob = self._predict_batch(left_crop, right_crop)

            left_prob = self.last_left_prob
            right_prob = self.last_right_prob

            # Draw rectangles
            left_color = (0, 255, 0) if (left_prob and left_prob > self.threshold) else (0, 0, 255)
            right_color = (0, 255, 0) if (right_prob and right_prob > self.threshold) else (0, 0, 255)

            cv2.rectangle(display, (left_box[0], left_box[1]), (left_box[2], left_box[3]), left_color, 2)
            cv2.rectangle(display, (right_box[0], right_box[1]), (right_box[2], right_box[3]), right_color, 2)

            # Show crops
            self._show_crop(left_crop, self.left_canvas)
            self._show_crop(right_crop, self.right_canvas)

            # Update labels
            self._update_eye_label(self.left_var, self.left_label, left_prob)
            self._update_eye_label(self.right_var, self.right_label, right_prob)
            self._update_status(left_prob, right_prob)
        else:
            self.status_var.set("No eyes detected")
            self.status_label.config(foreground="gray")
            self.left_var.set("--")
            self.right_var.set("--")

        # Zoom to face if detected
        if face_box:
            pad = int((face_box[2] - face_box[0]) * 0.3)
            fx1 = max(0, face_box[0] - pad)
            fy1 = max(0, face_box[1] - pad)
            fx2 = min(w, face_box[2] + pad)
            fy2 = min(h, face_box[3] + pad)
            display = display[fy1:fy2, fx1:fx2]

        self._show_frame(display)
        self.root.after(15, self._update)

    def _update_eye_label(self, var, label, prob):
        if prob is None:
            var.set("--")
            label.config(foreground="gray")
        elif prob > self.threshold:
            var.set(f"OPEN ({prob:.2f})")
            label.config(foreground="green")
        else:
            var.set(f"CLOSED ({prob:.2f})")
            label.config(foreground="red")

    def _update_status(self, left_prob, right_prob):
        if left_prob is None and right_prob is None:
            self.status_var.set("Detection failed")
            self.status_label.config(foreground="gray")
            return

        if left_prob is not None and right_prob is not None:
            avg = (left_prob + right_prob) / 2
        else:
            avg = left_prob if left_prob is not None else right_prob

        if avg > self.threshold:
            self.status_var.set("EYES OPEN")
            self.status_label.config(foreground="green")
        else:
            self.status_var.set("EYES CLOSED")
            self.status_label.config(foreground="red")

    def _show_crop(self, crop, canvas):
        if crop is None or crop.size == 0:
            return
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (80, 60))
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.delete("all")
        canvas.create_image(40, 30, image=img)
        canvas.image = img

    def _show_frame(self, frame):
        h, w = frame.shape[:2]
        scale = min(640 / w, 400 / h)
        new_size = (int(w * scale), int(h * scale))

        rgb = cv2.cvtColor(cv2.resize(frame, new_size), cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.video_canvas.delete("all")
        self.video_canvas.create_image(320, 200, image=img)
        self.video_canvas.image = img

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = EyeDetectorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
