from ultralytics import YOLO
import cv2
import os

# 🔹 Paths – CHANGE THESE
MODEL_PATH = r"E:\pytorch_deep_learning\test_muzzlemodel\best.pt"
IMAGE_PATH = r"E:\pytorch_deep_learning\test_muzzlemodel\test_cow1.jpg"
OUTPUT_PATH = r"E:\pytorch_deep_learning\test_muzzlemodel\output_muzzle.jpg"

# 1. Load model
model = YOLO(MODEL_PATH)

# 2. Read image with OpenCV
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

# 3. Run inference
results = model(IMAGE_PATH)[0]   # results is ultralytics.engine.results.Results

# 4. Get class names
names = model.names  # e.g. {0: 'muzzle'}

# 5. Loop over detections and draw only 'muzzle'
for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
    class_id = int(cls)
    label = names[class_id]

    # Only draw muzzle
    if label.lower() != "muzzle":
        continue

    x1, y1, x2, y2 = map(int, box)  # convert to int for cv2

    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Put label + confidence
    text = f"{label} {conf:.2f}"
    cv2.putText(img, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 6. Show result
cv2.imshow("Muzzle Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7. Save result
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
cv2.imwrite(OUTPUT_PATH, img)
print(f"Saved output image to: {OUTPUT_PATH}")
