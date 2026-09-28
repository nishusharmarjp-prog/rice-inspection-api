from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PIXELS_PER_MM = 10.0

@app.get("/")
def home():
    return {"status": "Server running successfully!"}

@app.post("/analyze")
async def analyze_rice(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    if white_pixels > total_pixels / 2:
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lengths = []
    widths = []
    broken_count = 0
    min_grain_area = 50

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_grain_area:
            continue

        rect = cv2.minAreaRect(cnt)
        w_px, h_px = rect[1]

        length_px = max(w_px, h_px)
        width_px = min(w_px, h_px)

        length_mm = length_px / PIXELS_PER_MM
        width_mm = width_px / PIXELS_PER_MM

        lengths.append(length_mm)
        widths.append(width_mm)

        if length_mm < 4.5:
            broken_count += 1
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(img, [box], 0, color, 2)

    total_grains = len(lengths)
    avg_len = round(float(np.mean(lengths)), 2) if total_grains > 0 else 0.0
    avg_wid = round(float(np.mean(widths)), 2) if total_grains > 0 else 0.0
    ratio = round(avg_len / avg_wid, 2) if avg_wid > 0 else 0.0
    broken_pct = round((broken_count / total_grains) * 100, 1) if total_grains > 0 else 0.0

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "total_grains": total_grains,
        "avg_length": avg_len,
        "avg_width": avg_wid,
        "lw_ratio": ratio,
        "broken_percentage": broken_pct,
        "image_base64": img_base64
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)