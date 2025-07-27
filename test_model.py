import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.transforms import functional as F
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the trained YOLOv10 model
model_path = r"C:\Users\bilal\OneDrive\Desktop\FYP\results\detect\train\weights\best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load YOLOv10 model using Ultralytics (auto-detects architecture)
model = YOLO(model_path)
model.to(device)
model.eval()

# Paths
test_images_folder = r"C:\Users\bilal\OneDrive\Desktop\FYP\New_data\test\images"
ground_truth_path = r"C:\Users\bilal\OneDrive\Desktop\FYP\New_data\test\labels"
output_folder = r"C:\Users\bilal\OneDrive\Desktop\FYP\test_results"
os.makedirs(output_folder, exist_ok=True)

# Class names
class_names = ["shelf", "void"]

# Performance Metrics Storage
all_preds = []
all_labels = []

# Process each test image
for img_name in os.listdir(test_images_folder):
    img_path = os.path.join(test_images_folder, img_name)
    image = cv2.imread(img_path)
    # Run inference
    with torch.no_grad():
        results = model(img_path)  # Pass image path to YOLOv10
    # Process detections
    detections = results[0].boxes.data.cpu().numpy()  # Get bounding box detections
    pred_classes = []  # Store detected class IDs for this image
    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]  # Extract bounding box, confidence, class index
        label = f"{class_names[int(cls)]}: {conf:.2f}"
        color = (0, 255, 0) if int(cls) == 1 else (0, 0, 255)  # Green for empty, Red for shelf
        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Store predictions
        pred_classes.append(int(cls))
    
    # Save detections for this image
    all_preds.extend(pred_classes)  # Extend to store multiple detections per image
    cv2.imwrite(os.path.join(output_folder, img_name), image)
    print(f"✅ Processed: {img_name} - {len(pred_classes)} objects detected")

    # Load ground truth labels for this image
    label_file = img_name.replace('.jpg', '.txt')
    label_path = os.path.join(ground_truth_path, label_file)
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                cls_id = int(line.split()[0])  # Extract class ID
                all_labels.append(cls_id)
    else:
        print(f"⚠️ Warning: Label file missing for {img_name}")

# After processing all images, check lengths
if len(all_preds) != len(all_labels):
    print(f"⚠️ Warning: Mismatch in predictions ({len(all_preds)}) and labels ({len(all_labels)})")
    # Decide how to handle this mismatch - truncate or pad one of the lists
    min_len = min(len(all_preds), len(all_labels))
    all_preds = all_preds[:min_len]
    all_labels = all_labels[:min_len]

# Compute performance metrics
if all_labels and all_preds:
    conf_matrix = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\n🔹 Confusion Matrix:\n", conf_matrix)
    print("\n🔹 Accuracy:", round(accuracy * 100, 2), "%")
    print("\n🔹 Classification Report:\n", class_report)
else:
    print("⚠️ Not enough data for evaluation - Check predictions and labels.")