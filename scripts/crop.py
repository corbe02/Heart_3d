import cv2
import os

# --- Config ---
input_video = "/home/corbe/heart_ws/src/heart_pkg/videos/HB_v1_src.mp4"       # video originale (1280x720)
output_video = "/home/corbe/heart_ws/src/heart_pkg/videos/HB_v1_cropped.mp4"       # video allineato (894x662)

# Dimensioni depth
depth_w, depth_h = 894, 662

# --- Apri il video originale ---
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise RuntimeError(f"Impossibile aprire il video {input_video}")

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video originale: {orig_w}x{orig_h} @ {fps} fps")

# Calcola l'offset per croppare centrato
dx = (orig_w - depth_w) // 2
dy = (orig_h - depth_h) // 2
print(f"Croppo: dx={dx}, dy={dy}, size=({depth_w}x{depth_h})")

# --- Setup writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (depth_w, depth_h))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop centrato
    crop = frame[dy:dy+depth_h, dx:dx+depth_w]

    out.write(crop)
    frame_idx += 1

cap.release()
out.release()
print(f"Video salvato in: {output_video}, totale frame: {frame_idx}")
