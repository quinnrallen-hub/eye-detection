# Eye Detection

Real-time eye open/closed detection using a custom-trained CNN (98% accuracy).

## Features

- Custom CNN trained from scratch on eye images
- No external ML models required (just OpenCV + PyTorch)
- Webcam support
- Live eye crop preview

## Requirements

```bash
pip install opencv-python numpy pillow torch requests
```

## Usage

```bash
python gui.py
```


### ESP32-CAM Setup

1. Flash `esp32cam/esp32cam.ino` to your ESP32-CAM
2. Update WiFi credentials in the sketch
3. Note the IP address from Serial Monitor
4. Enter the IP in the GUI and click "Connect ESP"

## How It Works

```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```

- EAR > 0.25 → Eyes open
- EAR < 0.20 → Eyes closed

## Files

- `gui.py` - Main GUI application
- `esp32cam/esp32cam.ino` - ESP32-CAM streaming firmware
