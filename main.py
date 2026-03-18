import cv2
import numpy as np
from ultralytics import YOLO

# 1. Setup Model and Camera
# Use Nano for fast segmentation that fits to the body silhouette
model = YOLO('yolov8n-seg.pt') 
cap = cv2.VideoCapture(0)

# Get dimensions
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 2. Initialize the Person-Mapped Memory (Accumulator)
# This float map stores 'heat' persistent only where the body *is*.
person_trace_map = np.zeros((h, w), dtype=np.float32)

# --- TUNING FOR "LIGHT RED" ---
DECAY_RATE = 0.90   # Medium persistence (0.1 to 0.99)
HEAT_GAIN = 10.0    # Lower gain so red doesn't saturate too fast
OVERLAY_OPACITY = 0.5 # Lower opacity (0.0 to 1.0) makes the red light/translucent
# ------------------------------

print("Starting. The person-mapped heatmap is active.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 3. Detect and Segment Persons
    results = model(frame, verbose=False, conf=0.5, stream=False)
    
    # Create an empty float mask for THIS frame (0.0 = background, 1.0 = person)
    current_person_mask = np.zeros((h, w), dtype=np.float32)

    # Filter for only 'Person' classes (Class 0)
    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().numpy()
        masks = results[0].masks.data.cpu().numpy() # (num_detections, h, w)
        person_masks = masks[clss == 0]

        # Combine all person masks into a single float map
        if len(person_masks) > 0:
            current_person_mask = np.any(person_masks, axis=0).astype(np.float32)

    # 4. Decay (Fading old positions)
    # The 'blue' trace only exists because this number is < 1.0.
    person_trace_map *= DECAY_RATE

    # 5. Accumulate (Adding heat only on current body pixels)
    # This prevents the heatmap from sticking to the background.
    person_trace_map += current_person_mask

    # 6. Visualization for "Light Red"
    # The crucial change: Normalizing and Brightening
    # Instead of just normalization, we add scale (alpha) and shift (beta)
    # This stretches the contrast and forces the map into a 'light' spectrum.
    acc_norm = cv2.normalize(person_trace_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # --- BRIGHTNESS BOOST ---
    # Multiplying by 1.5 increases contrast, +30 makes it lighter
    acc_norm = cv2.convertScaleAbs(acc_norm, alpha=1.5, beta=30)
    # ------------------------
    
    # Apply Colormap (JET: Blue -> Green -> Red)
    heatmap_color = cv2.applyColorMap(acc_norm, cv2.COLORMAP_JET)

    # 7. Final Composite
    # Desaturate background for better visibility
    bg_gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    
    # Composite the final frame using 'addWeighted'
    # We blend the background, the colorful heatmap, and the original color for reality.
    # The lower OVERLAY_OPACITY (0.5) is what makes the red light and translucent.
    output = cv2.addWeighted(bg_gray, 1 - OVERLAY_OPACITY, heatmap_color, OVERLAY_OPACITY, 0)
    # We add a small amount of original color back in so skin tones are visible through the red.
    output = cv2.addWeighted(output, 0.8, frame, 0.2, 0)

    # Show the result
    cv2.imshow('Dynamic Body-Mapped Trace Heatmap (Light Red)', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()