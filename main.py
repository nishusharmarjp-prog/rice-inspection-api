from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

app = FastAPI(title="Rice Grain Inspection Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Rice Inspection Engine Online"}

@app.post("/analyze")
async def analyze_rice(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Image decode failed"}

    # Standard scale width
    h, w = img.shape[:2]
    target_w = 1200
    scale = target_w / float(w)
    img = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # 1. Circle Detection for Reference Coin
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=35,
        minRadius=40,
        maxRadius=250
    )

    coin_mask = np.zeros(gray.shape, dtype=np.uint8)
    pixels_per_mm = None

    if circles is not None:
        circles = np.uint16(np.around(circles))
        best_circle = max(circles[0, :], key=lambda c: c[2])
        cx, cy, r = best_circle[0], best_circle[1], best_circle[2]
        diameter_px = float(r * 2)

        cv2.circle(coin_mask, (cx, cy), int(r * 1.12), 255, -1)

        # Standard 2001 ₹1 Coin = 25.0 mm; New ₹1 Coin = 20.0 mm
        coin_real_diameter_mm = 25.0 if diameter_px > 120 else 20.0
        pixels_per_mm = diameter_px / coin_real_diameter_mm

    if pixels_per_mm is None or pixels_per_mm <= 0:
        pixels_per_mm = 14.5

    # 2. Extract Rice Contours (Fine boundaries without eroding tips)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grain_mask = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([180, 95, 255]))
    
    # Remove Coin
    grain_mask = cv2.bitwise_and(grain_mask, grain_mask, mask=cv2.bitwise_not(coin_mask))

    # Gentle morphological cleaning (only 1 iteration to prevent shrinking grain tips)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grain_mask = cv2.morphologyEx(grain_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lengths = []
    widths = []
    broken_count = 0

    # Calibration compensation factor (compensates 1.15x for edge anti-aliasing / lens perspective)
    correction_factor = 1.15

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 60 or area > 8000:
            continue

        rect = cv2.minAreaRect(cnt)
        dim1, dim2 = rect[1]

        raw_len_px = max(dim1, dim2)
        raw_wid_px = min(dim1, dim2)

        # Adjusted accurate millimeter measurement
        length_mm = round((raw_len_px / pixels_per_mm) * correction_factor, 2)
        width_mm = round((raw_wid_px / pixels_per_mm) * correction_factor, 2)

        if 2.0 <= length_mm <= 14.0 and 0.6 <= width_mm <= 4.0:
            lengths.append(length_mm)
            widths.append(width_mm)
            if length_mm < 5.0:
                broken_count += 1

    total_grains = len(lengths)
    if total_grains > 0:
        avg_length = round(float(np.mean(lengths)), 2)
        avg_width = round(float(np.mean(widths)), 2)
        broken_pct = round((broken_count / total_grains) * 100.0, 1)
        lw_ratio = round(avg_length / (avg_width if avg_width > 0 else 1), 2)
    else:
        avg_length = 0.0
        avg_width = 0.0
        broken_pct = 0.0
        lw_ratio = 0.0

    return {
        "total_grains": total_grains,
        "avg_length": avg_length,
        "avg_length_mm": avg_length,
        "avg_width": avg_width,
        "avg_width_mm": avg_width,
        "length_width_ratio": lw_ratio,
        "broken_percentage": broken_pct,
        "calibration_used": f"{round(pixels_per_mm, 2)} px/mm"
    }
