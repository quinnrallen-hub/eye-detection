#!/usr/bin/env python3
"""
Eye open/closed detection GUI.
MediaPipe Face Landmarker blendshapes → reliability-weighted blink score → alert.

Signal pipeline:
  eyeBlinkLeft / eyeBlinkRight blendshapes  (0 = open, 1 = closed)
  → per-eye EMA smoothing  (slow to close, fast to open — blinks don't count)
  → reliability-weighted combination  (eye with more observed range gets more weight)
  → threshold → EYES OPEN / CLOSED status
  → 2-second sustained-closure timer → alert
"""

import tkinter as tk
import tkinter.messagebox as mb
from tkinter import ttk
from collections import deque
import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import time
import threading
import requests
from pathlib import Path

# ── Canvas ────────────────────────────────────────────────────────────────────
VIDEO_W, VIDEO_H = 640, 400
CROP_W,  CROP_H  = 120, 90

# ── MediaPipe ─────────────────────────────────────────────────────────────────
MP_MODEL_PATH = str(Path(__file__).parent / "face_landmarker.task")
MP_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                 "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")

BLINK_LEFT  = "eyeBlinkLeft"   # 0 = open, 1 = closed  (subject's left  = image-right)
BLINK_RIGHT = "eyeBlinkRight"  # 0 = open, 1 = closed  (subject's right = image-left)

# Landmark indices for eye crop boxes and face oval overlay
RIGHT_EYE_IDXS = [33,  7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_IDXS  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
FACE_OVAL_IDXS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
                  379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172,  58, 132,  93,
                  234, 127, 162,  21,  54, 103,  67, 109]

# ── EMA ───────────────────────────────────────────────────────────────────────
# Applied to the blink score (0=open → 1=closed).
# Slow to rise (closing): brief blinks don't push the smoothed score above threshold.
# Fast to fall  (opening): recovers immediately when eyes genuinely open.
BLINK_EMA_RISE = 0.25   # alpha when score is going up   (eyes closing)
BLINK_EMA_FALL = 0.80   # alpha when score is going down (eyes opening)

# ── Reliability ───────────────────────────────────────────────────────────────
# Each eye's weight = its observed range (max−min) over the history window.
# An eye whose blendshape never varies gets weight ≈ 0.
HIST_FRAMES = 90   # ~3 s at 30 fps

# ── Alert timing ──────────────────────────────────────────────────────────────
ALERT_AFTER_SECS     = 2.0
FACE_CANCEL_SECS     = 3.0

FPS_ALPHA = 0.1


def _blink_ema(new: float, old: float | None) -> float:
    if old is None:
        return new
    alpha = BLINK_EMA_RISE if new > old else BLINK_EMA_FALL
    return alpha * new + (1.0 - alpha) * old


class EyeDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Detector")
        self.root.geometry("800x680")

        # ── Camera / runtime ──────────────────────────────────────────────────
        self.cap            = None
        self.running        = False
        self.last_time      = time.time()
        self.fps            = 0.0

        # Blink threshold: eye is CLOSED when smoothed blink score > this value.
        # Exposed via slider (0–1). Default 0.35 works for most people.
        self.blink_threshold = 0.35

        # ── Detection state ───────────────────────────────────────────────────
        self.smooth_blink_l  = None   # smoothed blink score, subject's left eye
        self.smooth_blink_r  = None   # smoothed blink score, subject's right eye
        self._hist_l         = deque(maxlen=HIST_FRAMES)
        self._hist_r         = deque(maxlen=HIST_FRAMES)
        self.cached_eyes     = None
        self.last_face       = None

        # ── Alert state ───────────────────────────────────────────────────────
        self.closed_since    = None
        self.face_lost_since = None
        self.alert_active    = False

        # ── MediaPipe ─────────────────────────────────────────────────────────
        if not Path(MP_MODEL_PATH).exists():
            print("Downloading MediaPipe face landmarker model (~6 MB)…")
            urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_PATH)
        self.face_mesh = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL_PATH),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
            )
        )
        self._mp_ts = 0

        self._setup_ui()
        self._start_stream()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Camera row
        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=5)
        ttk.Label(top, text="Camera:").pack(side=tk.LEFT, padx=5)
        self.cam_var = tk.IntVar(value=0)
        ttk.Spinbox(top, from_=0, to=4, textvariable=self.cam_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Switch", command=self._switch_camera).pack(side=tk.LEFT, padx=5)

        # ESP32-CAM row
        esp = ttk.Frame(main)
        esp.pack(fill=tk.X, pady=2)
        ttk.Label(esp, text="ESP32-CAM URL:").pack(side=tk.LEFT, padx=5)
        self.cam_url_var = tk.StringVar(value="http://192.168.x.x:81/stream")
        ttk.Entry(esp, textvariable=self.cam_url_var, width=28).pack(side=tk.LEFT, padx=5)
        ttk.Button(esp, text="Use ESP32-CAM", command=self._connect_esp_cam).pack(side=tk.LEFT, padx=5)
        ttk.Label(esp, text="Alert IP:").pack(side=tk.LEFT, padx=15)
        self.alert_ip_var = tk.StringVar(value="192.168.251.214")
        ttk.Entry(esp, textvariable=self.alert_ip_var, width=16).pack(side=tk.LEFT, padx=5)

        # Video canvas
        self.video_canvas = tk.Canvas(main, width=VIDEO_W, height=VIDEO_H, bg="black")
        self.video_canvas.pack(pady=10)

        # Eye crops
        crops = ttk.Frame(main)
        crops.pack(pady=5)
        ttk.Label(crops, text="Left Eye:").pack(side=tk.LEFT, padx=5)
        self.left_canvas = tk.Canvas(crops, width=CROP_W, height=CROP_H, bg="gray",
                                     highlightthickness=2, highlightbackground="gray")
        self.left_canvas.pack(side=tk.LEFT, padx=5)
        ttk.Label(crops, text="Right Eye:").pack(side=tk.LEFT, padx=20)
        self.right_canvas = tk.Canvas(crops, width=CROP_W, height=CROP_H, bg="gray",
                                      highlightthickness=2, highlightbackground="gray")
        self.right_canvas.pack(side=tk.LEFT, padx=5)

        # Status labels
        labels = ttk.Frame(main)
        labels.pack(pady=10)
        self.left_var   = tk.StringVar(value="--")
        self.status_var = tk.StringVar(value="Starting…")
        self.right_var  = tk.StringVar(value="--")
        self.left_label   = ttk.Label(labels, textvariable=self.left_var,   font=("Helvetica", 16, "bold"))
        self.status_label = ttk.Label(labels, textvariable=self.status_var, font=("Helvetica", 22, "bold"))
        self.right_label  = ttk.Label(labels, textvariable=self.right_var,  font=("Helvetica", 16, "bold"))
        self.left_label.pack(side=tk.LEFT, padx=30)
        self.status_label.pack(side=tk.LEFT, padx=30)
        self.right_label.pack(side=tk.LEFT, padx=30)

        # Threshold slider  (controls blink score threshold, 0–1)
        tf = ttk.Frame(main)
        tf.pack(pady=10, fill=tk.X, padx=50)
        ttk.Label(tf, text="Sensitivity:").pack(side=tk.LEFT, padx=5)
        self.thresh_var = tk.DoubleVar(value=self.blink_threshold)
        ttk.Scale(tf, from_=0.0, to=1.0, variable=self.thresh_var,
                  orient=tk.HORIZONTAL,
                  command=self._on_threshold_change).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.thresh_label = ttk.Label(tf, text=f"{self.blink_threshold:.2f}", width=5)
        self.thresh_label.pack(side=tk.LEFT, padx=5)
        ttk.Label(tf, text="← more sensitive    less sensitive →",
                  foreground="gray").pack(side=tk.LEFT, padx=8)

    # ── Camera helpers ────────────────────────────────────────────────────────

    def _reset_state(self):
        self.smooth_blink_l  = None
        self.smooth_blink_r  = None
        self._hist_l.clear()
        self._hist_r.clear()
        self.cached_eyes     = None
        self.closed_since    = None
        self.face_lost_since = None
        self._mp_ts          = 0
        if self.alert_active:
            self.alert_active = False
            self._send_alert(False)

    def _open_camera(self, source):
        if self.cap:
            self.cap.release()
        self._reset_state()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.status_var.set("Cannot open camera")
            self.status_label.config(foreground="red")
            return
        if not self.running:
            self.running = True
            self._update()

    def _connect_esp_cam(self):
        url = self.cam_url_var.get().strip()
        if not url or url == "http://192.168.x.x:81/stream":
            self.status_var.set("Enter ESP32-CAM URL first")
            self.status_label.config(foreground="orange")
            return
        self._open_camera(url)

    def _switch_camera(self):
        self._open_camera(self.cam_var.get())

    def _start_stream(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.running = True
            self._update()
        else:
            self.status_var.set("No camera found")

    def _on_threshold_change(self, value):
        self.blink_threshold = float(value)
        self.thresh_label.config(text=f"{self.blink_threshold:.2f}")

    # ── Alert ─────────────────────────────────────────────────────────────────

    def _send_alert(self, on: bool):
        ip = self.alert_ip_var.get().strip()
        if not ip:
            return
        url = f"http://{ip}/alert/{'on' if on else 'off'}"
        threading.Thread(
            target=lambda: requests.get(url, timeout=1),
            daemon=True
        ).start()

    # ── MediaPipe ─────────────────────────────────────────────────────────────

    def _run_mediapipe(self, frame):
        """Return (eye_boxes, blink_l, blink_r) or ([], None, None) if no face."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._mp_ts += 1
        results = self.face_mesh.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            self._mp_ts,
        )

        if not results.face_landmarks or not results.face_blendshapes:
            self.last_face   = None
            self.cached_eyes = None
            return [], None, None

        lm = results.face_landmarks[0]

        # Blink scores
        blink_l = blink_r = None
        for cat in results.face_blendshapes[0]:
            if   cat.category_name == BLINK_LEFT:  blink_l = cat.score
            elif cat.category_name == BLINK_RIGHT:  blink_r = cat.score

        # Face oval for overlay
        oxs = [lm[i].x * w for i in FACE_OVAL_IDXS]
        oys = [lm[i].y * h for i in FACE_OVAL_IDXS]
        fx, fy = int(min(oxs)), int(min(oys))
        self.last_face = (fx, fy, int(max(oxs)) - fx, int(max(oys)) - fy)

        # Eye crop boxes  (generous padding so the CNN can be retrained later)
        def _box(idxs, pad=0.6):
            xs = [lm[i].x * w for i in idxs]
            ys = [lm[i].y * h for i in idxs]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            ew, eh = x2 - x1, y2 - y1
            return (max(0, int(x1 - ew * pad)), max(0, int(y1 - eh * pad)),
                    min(w, int(x2 + ew * pad)), min(h, int(y2 + eh * pad)))

        # eye_boxes[0] = subject's right (image-left)  → right_box in _update
        # eye_boxes[1] = subject's left  (image-right) → left_box  in _update
        new_eyes = [_box(RIGHT_EYE_IDXS), _box(LEFT_EYE_IDXS)]
        EMA = 0.3
        if self.cached_eyes is None:
            self.cached_eyes = new_eyes
        else:
            self.cached_eyes = [
                tuple(int(EMA * n + (1 - EMA) * o) for n, o in zip(ne, oe))
                for ne, oe in zip(new_eyes, self.cached_eyes)
            ]

        return self.cached_eyes, blink_l, blink_r

    # ── Reliability weighting ────────────────────────────────────────────────

    def _weights(self):
        """Weight each eye by its observed blink-score range. Falls back to 0.5/0.5."""
        if len(self._hist_l) < 10:
            return 0.5, 0.5
        rl = max(self._hist_l) - min(self._hist_l)
        rr = max(self._hist_r) - min(self._hist_r)
        total = rl + rr
        if total < 1e-4:          # both stuck — equal weight
            return 0.5, 0.5
        return rl / total, rr / total

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _corner_box(self, img, x1, y1, x2, y2, color, thick=2, arm=14):
        for sx, sy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(img, (sx, sy), (sx + dx * arm, sy),        color, thick)
            cv2.line(img, (sx, sy), (sx, sy + dy * arm),        color, thick)

    def _show_crop(self, crop, canvas):
        if crop is None or crop.size == 0:
            return
        rgb = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (CROP_W, CROP_H))
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.delete("all")
        canvas.create_image(CROP_W // 2, CROP_H // 2, image=img)
        canvas.image = img

    def _show_frame(self, frame):
        h, w = frame.shape[:2]
        scale = min(VIDEO_W / w, VIDEO_H / h)
        rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                         (int(w * scale), int(h * scale)))
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_canvas.delete("all")
        self.video_canvas.create_image(VIDEO_W // 2, VIDEO_H // 2, image=img)
        self.video_canvas.image = img

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _update(self):
        if not self.running:
            return
        if not self.cap or not self.cap.isOpened():
            self.root.after(30, self._update)
            return

        try:
            ret, frame = self.cap.read()
            if not ret:
                self.status_var.set("Camera read failed")
                self.status_label.config(foreground="gray")
                return

            now = time.time()
            self.fps = FPS_ALPHA * (1.0 / max(now - self.last_time, 0.001)) + (1 - FPS_ALPHA) * self.fps
            self.last_time = now

            display = frame.copy()
            eye_boxes, blink_l, blink_r = self._run_mediapipe(frame)

            if len(eye_boxes) == 2:
                self.face_lost_since = None
                right_box, left_box  = eye_boxes[0], eye_boxes[1]

                right_crop = frame[right_box[1]:right_box[3], right_box[0]:right_box[2]]
                left_crop  = frame[left_box[1]:left_box[3],   left_box[0]:left_box[2]]

                # Smooth each eye's blink score independently
                if blink_l is not None:
                    self.smooth_blink_l = _blink_ema(blink_l, self.smooth_blink_l)
                    self._hist_l.append(self.smooth_blink_l)
                if blink_r is not None:
                    self.smooth_blink_r = _blink_ema(blink_r, self.smooth_blink_r)
                    self._hist_r.append(self.smooth_blink_r)

                # Reliability-weighted combined blink score
                w_l, w_r = self._weights()
                bl = self.smooth_blink_l or 0.0
                br = self.smooth_blink_r or 0.0
                combined_blink = w_l * bl + w_r * br

                # Per-eye open/closed for labels and box colour
                l_closed = bl > self.blink_threshold
                r_closed = br > self.blink_threshold
                eyes_closed = combined_blink > self.blink_threshold

                left_color  = (0, 0, 255) if l_closed else (0, 255, 0)
                right_color = (0, 0, 255) if r_closed else (0, 255, 0)

                # Face overlay
                if self.last_face:
                    fx, fy, fw, fh = self.last_face
                    self._corner_box(display, fx, fy, fx+fw, fy+fh, (200,200,200), 1, 16)

                self._corner_box(display, *left_box,  left_color)
                self._corner_box(display, *right_box, right_color)

                self._show_crop(left_crop,  self.left_canvas)
                self._show_crop(right_crop, self.right_canvas)
                self.left_canvas.config(highlightbackground="red" if l_closed else "green")
                self.right_canvas.config(highlightbackground="red" if r_closed else "green")

                # Labels: show blink score so user can see raw values
                self.left_var.set(f"{'CLOSED' if l_closed else 'OPEN'} ({bl:.2f})")
                self.left_label.config(foreground="red" if l_closed else "green")
                self.right_var.set(f"{'CLOSED' if r_closed else 'OPEN'} ({br:.2f})")
                self.right_label.config(foreground="red" if r_closed else "green")

                self._update_status(eyes_closed)

            else:
                # No face
                self.smooth_blink_l = self.smooth_blink_r = None
                self.status_var.set("No face detected")
                self.status_label.config(foreground="gray")
                self.left_var.set("--");  self.right_var.set("--")
                self.left_canvas.config(highlightbackground="gray")
                self.right_canvas.config(highlightbackground="gray")

                if self.face_lost_since is None:
                    self.face_lost_since = time.time()
                if time.time() - self.face_lost_since > FACE_CANCEL_SECS:
                    self.closed_since = None
                    if self.alert_active:
                        self.alert_active = False
                        self._send_alert(False)

            # Debug overlay
            w_l, w_r = self._weights()
            lines = [
                f"FPS {self.fps:.0f}",
                f"blink L:{self.smooth_blink_l:.2f}  R:{self.smooth_blink_r:.2f}"
                    if self.smooth_blink_l is not None else "blink --",
                f"weight L:{w_l:.2f}  R:{w_r:.2f}",
            ]
            for i, ln in enumerate(lines):
                cv2.putText(display, ln, (8, 22 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            self._show_frame(display)

        except Exception as e:
            print(f"Frame error: {e}")
        finally:
            self.root.after(15, self._update)

    def _update_status(self, eyes_closed: bool):
        if not eyes_closed:
            self.status_var.set("EYES OPEN")
            self.status_label.config(foreground="green")
            self.closed_since = None
            if self.alert_active:
                self.alert_active = False
                self._send_alert(False)
        else:
            if self.closed_since is None:
                self.closed_since = time.time()
            secs = time.time() - self.closed_since
            if secs >= ALERT_AFTER_SECS:
                self.status_var.set(f"EYES CLOSED ({int(secs)}s)")
                if not self.alert_active:
                    self.alert_active = True
                    self._send_alert(True)
            else:
                self.status_var.set("EYES CLOSED")
            self.status_label.config(foreground="red")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def on_close(self):
        self.running = False
        if self.alert_active:
            self._send_alert(False)
        if self.cap:
            self.cap.release()
        self.face_mesh.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        app = EyeDetectorGUI(root)
    except Exception as e:
        mb.showerror("Startup Error", str(e))
        root.destroy()
        return
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
