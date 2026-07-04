import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\Users\Reddy\Documents\Minor Project\runs\detect\train2\weights\best.pt"
IMAGE_PATH = r"C:\Users\Reddy\Documents\Minor Project\Datasets\PDT dataset\LL\YOLO_txt\test\images\images_(13)_0036.jpg"
OUTPUT_DIR = r"C:\Users\Reddy\Documents\Minor Project\runs\detect\gradcam_images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv8 model
yolo = YOLO(MODEL_PATH)
model = yolo.model.to(DEVICE)
model.eval()

# Hook storage
activations = []
gradients = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

# Target layer (last conv before Detect)
target_layer = model.model[-2]

# Register hooks
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# Load image
img_bgr = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (640, 640))
img_norm = img_resized.astype(np.float32) / 255.0

# Tensor
x = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
x.requires_grad = True

# Forward pass
preds = model(x)

# ✅ Select ONE detection score for backprop
# preds[0] shape: [num_preds, 6] → (x,y,w,h,conf,class)
score = preds[0][:, 4].max()  # highest confidence

# Backward pass
model.zero_grad()
score.backward()

# Get CAM
act = activations[0].detach()
grad = gradients[0].detach()

weights = grad.mean(dim=(2, 3), keepdim=True)
cam = (weights * act).sum(dim=1).squeeze()
cam = torch.relu(cam)

cam = cam.cpu().numpy()
cam = cv2.resize(cam, (640, 640))
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# Heatmap
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)

# Save
out_path = os.path.join(OUTPUT_DIR, "gradcam_images_(13)_0036.jpg")
cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print("✅ YOLO Grad-CAM saved at:")
print(out_path)
