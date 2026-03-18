import cv2
import numpy as np
from ultralytics import YOLO

def main():
    model = YOLO('yolov8n-seg.pt') 
    cap = cv2.VideoCapture(0)
    w, h = int(cap.get(3)), int(cap.get(4))

    # Heat Memory
    heatmap_memory = np.zeros((h, w), dtype=np.float32)

    # --- TUNED TO REMOVE BLUE LATENCY ---
    DECAY_RATE = 0.70   # Lower = Faster disappearance of blue traces
    HEAT_GAIN = 50.0    # Higher = Faster turn to Red
    BLUR_SIZE = 31      # Slightly smaller blur for a tighter glow
    # ------------------------------------

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. FAST DECAY: This clears the blue 'ghosts' instantly
        heatmap_memory *= DECAY_RATE

        # 2. AI SEGMENTATION
        results = model(frame, verbose=False, conf=0.5, classes=[0])

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            combined_mask = np.any(masks, axis=0).astype(np.float32)
            
            # 3. INJECT HEAT: High gain for instant response
            heatmap_memory += (combined_mask * HEAT_GAIN)

        # 4. PROCESSING
        heatmap_memory = np.clip(heatmap_memory, 0, 255)
        
        # Apply blur for the "Presence Probability" look
        blurred_heat = cv2.GaussianBlur(heatmap_memory, (BLUR_SIZE, BLUR_SIZE), 0)
        
        # Colorize
        heat_norm = blurred_heat.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

        # 5. OVERLAY (Darken background to show glow)
        bg_dimmed = cv2.convertScaleAbs(frame, alpha=0.6, beta=0)
        mask = heat_norm > 15 # Ignore very faint blue noise
        
        output = bg_dimmed.copy()
        output[mask] = cv2.addWeighted(bg_dimmed, 0.3, heatmap_color, 0.7, 0)[mask]

        cv2.imshow('No-Latency Intervention Heatmap', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()