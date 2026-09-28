from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

app = FastAPI(title="Rice Grain Quality Inspection Engine")

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
    # 1. Read Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Image decode failed"}

    # Resize for processing if image is too large while keeping aspect ratio
    h, w = img.shape[:2]
    max_dim = 1600
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2. Adaptive / Otsu Thresholding for clean grain contours
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Invert agar background dark sheet ho
    white_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    if white_pixels > total_pixels / 2:
        thresh = cv2.bitwise_not(thresh)

    # Morphological cleaning: Remove shadows and separate slightly touching grains
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Reference Calibration (Default fallback: fixed DPI reference if coin is not found)
    # Default: ~12 pixels per mm (phone camera at 15-20cm standard height)
    pixels_per_mm = 12.0  
    
    # Check for calibration coin (Circular object with large area)
    grain_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # Ignore tiny dust noise
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # Sikka detection (Circle area > 3500 and circularity > 0.75)
        if area > 3500 and circularity > 0.75:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            diameter_px = radius * 2.0
            # 1 Rupee coin diameter = 20 mm
            pixels_per_mm = diameter_px / 20.0
        else:
            grain_contours.append(cnt)

    # 4. Measure Each Grain
    lengths = []
    widths = []
    broken_count = 0

    for cnt in grain_contours:
        area = cv2.contourArea(cnt)
        # Chawal ke single daane ka normal area filter (dust filter)
        if area < 80:
            continue

        # Minimum bounding rotated rectangle
        rect = cv2.minAreaRect(cnt)
        dim1, dim2 = rect[1]
        
        # Length hamesha badi dimension hoti hai, width chhoti
        length_px = max(dim1, dim2)
        width_px = min(dim1, dim2)

        length_mm = round(length_px / pixels_per_mm, 2)
        width_mm = round(width_px / pixels_per_mm, 2)

        # Basic grain sanity filter (Rice length generally between 2mm to 11mm)
        if 1.5 <= length_mm <= 12.0 and 0.5 <= width_mm <= 4.0:
            lengths.append(length_mm)
            widths.append(width_mm)
            
            # Agar length 4.5 mm se kam ho to use Broken maana jata hai
            if length_mm < 4.5:
                broken_count += 1

    total_grains = len(lengths)
    if total_grains > 0:
        avg_length = round(float(np.mean(lengths)), 2)
        avg_width = round(float(np.mean(widths)), 2)
        broken_pct = round((broken_count / total_grains) * 100.0, 1)
        # Length to Width ratio (Basmati standard metric)
        lw_ratio = round(avg_length / (avg_width if avg_width > 0 else 1), 2)
    else:
        avg_length = 0.0
        avg_width = 0.0
        broken_pct = 0.0
        lw_ratio = 0.0

    return {
        "total_grains": total_grains,
        "avg_length_mm": avg_length,
        "avg_width_mm": avg_width,
        "length_width_ratio": lw_ratio,
        "broken_percentage": broken_pct,
        "calibration_used": f"{round(pixels_per_mm, 2)} px/mm"
    }
