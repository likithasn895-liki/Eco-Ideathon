# Inference pipeline placeholder
import cv2
import json
import math
import pandas as pd
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

# -----------------------------
# Load model
# -----------------------------
model = YOLO("solar-panels-1/yolov8n.pt")

# -----------------------------
# Helper: compute area of polygon
# -----------------------------
def polygon_area(points):
    poly = Polygon(points)
    return poly.area

# -----------------------------
# Helper: compute overlap with buffer circle
# -----------------------------
def overlap_area(lat, lon, radius, box):
    x1, y1, x2, y2 = box
    panel_poly = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    circle = Point(lat, lon).buffer(radius)
    return panel_poly.intersection(circle).area

# -----------------------------
# Main inference
# -----------------------------
def process_row(sample_id, lat, lon, image_path):
    img = cv2.imread(image_path)
    results = model(img)[0]

    boxes = results.boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        return {
            "sample_id": sample_id,
            "lat": lat,
            "lon": lon,
            "has_solar": False,
            "buffer_radius": 2400,
            "total_panel_area": 0
        }

    # First try 1200
    radius = 1200
    best_overlap = 0
    best_box = None

    for box in boxes:
        ov = overlap_area(lat, lon, radius, box)
        if ov > best_overlap:
            best_overlap = ov
            best_box = box

    if best_overlap == 0:
        # Try 2400
        radius = 2400
        for box in boxes:
            ov = overlap_area(lat, lon, radius, box)
            if ov > best_overlap:
                best_overlap = ov
                best_box = box

        if best_overlap == 0:
            return {
                "sample_id": sample_id,
                "lat": lat,
                "lon": lon,
                "has_solar": False,
                "buffer_radius": 2400,
                "total_panel_area": 0
            }

    # Compute panel area (simple bbox area)
    x1, y1, x2, y2 = best_box
    panel_area = abs((x2 - x1) * (y2 - y1))

    # Save overlay
    cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
    cv2.imwrite(f"output/{sample_id}_overlay.jpg", img)

    return {
        "sample_id": sample_id,
        "lat": lat,
        "lon": lon,
        "has_solar": True,
        "buffer_radius": radius,
        "total_panel_area": float(panel_area)
    }

# -----------------------------
# Run on Excel
# -----------------------------
df = pd.read_excel("input/input.xlsx")

outputs = []

for _, row in df.iterrows():
    out = process_row(
        row["sample_id"],
        row["latitude"],
        row["longitude"],
        row["image_path"]
    )
    outputs.append(out)

with open("output/results.json", "w") as f:
    json.dump(outputs, f, indent=4)

print("DONE")
