#!/usr/bin/env python3
"""
Eye open/closed detection GUI.
Uses MediaPipe FaceLandmarker to find eyes, CNN to classify open/closed.
"""

import tkinter as tk
from tkinter import ttk
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp

# --- Config ---
IMG_SIZE = 64
MODEL_PATH = "/home/quinn/eye_detection_project/eye_classifier.pth"
LANDMARKER_PATH = "/home/quinn/eye_detection_project/face_landmarker.task"

# ESP32-CAM stream URL (change IP to match your camera)
ESPCAM_URL = "http://192.168.251.119:81/stream"

# MediaPipe landmark indices (from subject's perspective)
# Left eye = 362-398 range, Right eye = 33-173 range
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249,
                    263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155,
                     133, 173, 157, 158, 159, 160, 161, 246]

# 6-point subsets for EAR: [p1, p2, p3, p4, p5, p6]
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.22
PADDING = 0.4


class EyeCNN(nn.Module):
    """Simple CNN for eye open/closed classification."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class EyeDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Open/Closed Detector")
        self.root.geometry("800x700")

        # State
        self.cap = None
        self.running = False
        self.current_cam = 0
        self.available_cameras = []
        self.threshold = 0.5

        # Load models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.landmarker = self._load_landmarker()

        # Build UI
        self._setup_ui()

        # Start
        self._detect_cameras()
        self._start_stream()

    def _load_model(self):
        model = EyeCNN().to(self.device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
        model.eval()
        return model

    def _load_landmarker(self):
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=LANDMARKER_PATH),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def _setup_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Camera selection
        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=5)

        ttk.Label(top, text="Camera:").pack(side=tk.LEFT, padx=5)
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(top, textvariable=self.cam_var, state="readonly", width=12)
        self.cam_combo.pack(side=tk.LEFT, padx=5)
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_camera_change)
        ttk.Button(top, text="Refresh", command=self._detect_cameras).pack(side=tk.LEFT, padx=5)

        # ESP URL input
        ttk.Label(top, text="ESP IP:").pack(side=tk.LEFT, padx=(15, 5))
        self.esp_ip_var = tk.StringVar(value="192.168.251.119")
        self.esp_ip_entry = ttk.Entry(top, textvariable=self.esp_ip_var, width=15)
        self.esp_ip_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Connect ESP", command=self._connect_esp).pack(side=tk.LEFT, padx=5)

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

        # EAR and brightness display
        info_frame = ttk.Frame(main)
        info_frame.pack(pady=2)

        self.ear_var = tk.StringVar(value="EAR: --")
        ttk.Label(info_frame, textvariable=self.ear_var, font=("Helvetica", 11)).pack(side=tk.LEFT, padx=20)

        self.bright_var = tk.StringVar(value="Brightness: --")
        ttk.Label(info_frame, textvariable=self.bright_var, font=("Helvetica", 11)).pack(side=tk.LEFT, padx=20)

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

        ttk.Button(thresh_frame, text="Auto", command=self._auto_threshold_btn).pack(side=tk.LEFT, padx=10)

        self.auto_thresh_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(thresh_frame, text="Continuous", variable=self.auto_thresh_var).pack(side=tk.LEFT)

    def _detect_cameras(self):
        self.available_cameras = []

        # Check local cameras
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.available_cameras.append(("local", i, f"Camera {i}"))
                cap.release()

        # Add ESP32-CAM option
        self.available_cameras.append(("esp", ESPCAM_URL, "ESP32-CAM"))

        if self.available_cameras:
            self.cam_combo["values"] = [c[2] for c in self.available_cameras]
            self.cam_combo.current(0)
        else:
            self.cam_combo["values"] = ["No cameras"]
            self.cam_combo.current(0)
            self.status_var.set("No cameras found")

    def _start_stream(self):
        if not self.available_cameras:
            return

        self.current_cam = self.available_cameras[0]
        self._open_camera(self.current_cam)

        if self.cap and self.cap.isOpened():
            self.running = True
            self._update()
        else:
            self.status_var.set("Failed to open camera")

    def _open_camera(self, cam_info):
        """Open camera by type (local index or ESP URL)."""
        if self.cap:
            self.cap.release()

        cam_type, cam_source, _ = cam_info
        if cam_type == "local":
            self.cap = cv2.VideoCapture(cam_source)
        else:  # esp
            self.cap = cv2.VideoCapture(cam_source)
            # Set buffer size low for lower latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _on_camera_change(self, event=None):
        idx = self.cam_combo.current()
        if 0 <= idx < len(self.available_cameras):
            new_cam = self.available_cameras[idx]
            if new_cam != self.current_cam:
                self.current_cam = new_cam
                self._open_camera(new_cam)

    def _on_threshold_change(self, value):
        self.threshold = float(value)
        self.thresh_label.config(text=f"{self.threshold:.2f}")

    def _connect_esp(self):
        """Connect to ESP32-CAM using IP from entry field."""
        ip = self.esp_ip_var.get().strip()
        if not ip:
            self.status_var.set("Enter ESP IP address")
            return

        url = f"http://{ip}:81/stream"
        self.status_var.set(f"Connecting to {ip}...")
        self.root.update()

        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.cap.isOpened():
            ret, _ = self.cap.read()
            if ret:
                self.current_cam = ("esp", url, f"ESP@{ip}")
                self.status_var.set(f"Connected to ESP@{ip}")
                if not self.running:
                    self.running = True
                    self._update()
            else:
                self.status_var.set(f"ESP at {ip} not streaming")
        else:
            self.status_var.set(f"Failed to connect to {ip}")

    def _compute_ear(self, landmarks, indices):
        """Eye Aspect Ratio = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)"""
        def dist(a, b):
            return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

        pts = [landmarks[i] for i in indices]
        v1 = dist(pts[1], pts[5])
        v2 = dist(pts[2], pts[4])
        h = dist(pts[0], pts[3])

        return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.0

    def _crop_eye(self, frame, landmarks, indices):
        """Crop eye region with padding, return crop or None."""
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

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None, (0, 0, 0, 0)

        return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    def _predict_ear(self, ear_value):
        """Use EAR to determine if eye is open. Returns open probability."""
        # EAR > 0.22 typically means open, < 0.22 means closed
        # Convert to a probability-like value
        if ear_value > 0.30:
            return 1.0
        elif ear_value < 0.15:
            return 0.0
        else:
            # Linear interpolation between 0.15 and 0.30
            return (ear_value - 0.15) / 0.15

    def _get_brightness(self, frame):
        """Get average brightness of frame (0-255)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    def _auto_threshold(self, frame):
        """Adjust threshold based on frame brightness."""
        brightness = self._get_brightness(frame)

        # Bright (>150): use lower threshold (easier to detect)
        # Dark (<80): use higher threshold (harder to detect)
        # Map brightness 50-200 to threshold 0.6-0.4
        if brightness > 200:
            new_thresh = 0.40
        elif brightness < 50:
            new_thresh = 0.65
        else:
            # Linear interpolation
            new_thresh = 0.65 - (brightness - 50) * (0.25 / 150)

        self.threshold = new_thresh
        self.thresh_var.set(new_thresh)
        self.thresh_label.config(text=f"{new_thresh:.2f}")

    def _auto_threshold_btn(self):
        """Button callback for auto threshold."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self._auto_threshold(frame)


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

        # Update brightness display and auto-adjust threshold if enabled
        brightness = self._get_brightness(frame)
        self.bright_var.set(f"Brightness: {brightness:.0f}")

        if self.auto_thresh_var.get():
            self._auto_threshold(frame)

        h, w = frame.shape[:2]
        display = frame.copy()

        # Detect face
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)

        left_prob, right_prob = None, None

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            # EAR - use this for detection (more reliable than CNN)
            left_ear = self._compute_ear(lm, LEFT_EYE_EAR)
            right_ear = self._compute_ear(lm, RIGHT_EYE_EAR)
            self.ear_var.set(f"EAR: L={left_ear:.2f} R={right_ear:.2f}")

            # Use EAR for open/closed detection
            left_prob = self._predict_ear(left_ear)
            right_prob = self._predict_ear(right_ear)

            # Crop and display eyes
            left_crop, left_box = self._crop_eye(frame, lm, LEFT_EYE_CONTOUR)
            if left_crop is not None:
                cv2.rectangle(display, (left_box[0], left_box[1]), (left_box[2], left_box[3]), (0, 255, 0), 2)
                self._show_crop(left_crop, self.left_canvas)

            right_crop, right_box = self._crop_eye(frame, lm, RIGHT_EYE_CONTOUR)
            if right_crop is not None:
                cv2.rectangle(display, (right_box[0], right_box[1]), (right_box[2], right_box[3]), (0, 255, 0), 2)
                self._show_crop(right_crop, self.right_canvas)

            # Update labels
            self._update_eye_label(self.left_var, self.left_label, left_prob)
            self._update_eye_label(self.right_var, self.right_label, right_prob)
            self._update_status(left_prob, right_prob)
        else:
            self.status_var.set("No face detected")
            self.status_label.config(foreground="gray")
            self.ear_var.set("EAR: --")
            self.left_var.set("--")
            self.right_var.set("--")

        self._show_frame(display)
        self.root.after(30, self._update)

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

        # Use average or single eye
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
        if hasattr(self, "landmarker"):
            self.landmarker.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = EyeDetectorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
