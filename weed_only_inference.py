from ultralytics import YOLO
import cv2
import os

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\Users\Reddy\Documents\Minor Project\runs\detect\cwc_yolov8s_e50_refine\weights\best.pt"
SOURCE = r"C:\Users\Reddy\Documents\Minor Project\Datasets\CWC dataset\YOLO_txt\val\images\sedge__(11).jpg"   # image / folder / video
OUTPUT_DIR = r"C:\Users\Reddy\Documents\Minor Project\runs\detect\weed_only_output"
CONF_THRES = 0.4
# ----------------------------------------

# Weed classes
WEED_CLASSES = {
    "bluegrass",
    "chenopodium_album",
    "cirsium_setosum",
    "sedge",
    "nightshade",
    "velvet"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Run prediction
results = model(
    source=SOURCE,
    conf=CONF_THRES,
    device=0,
    stream=True
)

for r in results:
    img = r.orig_img.copy()
    names = r.names

    if r.boxes is None:
        continue

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = names[cls_id]

        # Decide label
        if cls_name in WEED_CLASSES:
            display_label = cls_name
            color = (0, 0, 255)   # red for weed
        else:
            display_label = "CROP"
            color = (0, 255, 0)   # green for crop

        # Bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{display_label} {conf:.2f}"

        # Text size
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Keep label inside image
        text_y = y1 - 10 if y1 - 10 > th else y1 + th + 10

        # Draw filled rectangle for label
        cv2.rectangle(
            img,
            (x1, text_y - th - 4),
            (x1 + tw + 4, text_y),
            color,
            -1
        )

        # Put text
        cv2.putText(
            img,
            label,
            (x1 + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # Save output
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(r.path))
    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")

print("✅ Weed-only inference completed.")
