import cv2
import numpy as np
from ultralytics import YOLO

def main():
    model = YOLO('yolov8n-seg.pt') 
    cap = cv2.VideoCapture(0)
    w, h = int(cap.get(3)), int(cap.get(4))

    # Heat Accumulator
    heatmap_memory = np.zeros((h, w), dtype=np.float32)

    # Parameters
    DECAY_RATE = 0.75
    HEAT_GAIN = 120.0  # High gain = Faster Red
    
    print("Running... Move to see the Red heat.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Decay the old heat
        heatmap_memory *= DECAY_RATE

        # 2. AI Segmentation
        results = model(frame, verbose=False, conf=0.5, classes=[0])

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            combined_mask = np.any(masks, axis=0).astype(np.float32)
            
            # 3. Inject Massive Heat
            heatmap_memory += (combined_mask * HEAT_GAIN)

        # 4. CAP AT 255 (This is the Red Threshold)
        heatmap_memory = np.clip(heatmap_memory, 0, 255)
        
        # 5. Apply Glow
        blurred_heat = cv2.GaussianBlur(heatmap_memory, (41, 41), 0)
        
        # 6. Final Color Mapping
        heat_norm = blurred_heat.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

        # Background Dimming (0.4 alpha makes colors pop)
        bg_dimmed = cv2.convertScaleAbs(frame, alpha=0.4, beta=0)
        
        # Mask to only show the person's glow
        mask = heat_norm > 15
        output = bg_dimmed.copy()
        
        # Overlay the Red/Blue heatmap
        output[mask] = cv2.addWeighted(bg_dimmed, 0.2, heatmap_color, 0.8, 0)[mask]

        # Add Legend
        cv2.putText(output, "INTERVENTION: RED (HIGH) / BLUE (LOW)", (20, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Red-Point Intervention Detection', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()