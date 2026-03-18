import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # 1. Load the AI Model
    model = YOLO('yolov8n-seg.pt') 

    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 2. This is our 'Canvas' for the Heat
    heatmap_memory = np.zeros((h, w), dtype=np.float32)

    # --- TUNED FOR THE IMAGE LOOK ---
    DECAY_RATE = 0.96   # High decay means the trail stays longer (smeared look)
    HEAT_GAIN = 8.0     # Moderate gain so it turns red slowly as you stay still
    BLUR_SIZE = 51      # This creates the "glow" around the body
    BG_BRIGHTNESS = 0.5 # Dims the background like in your photo
    # ---------------------------------

    while True:
        ret, frame = cap.read()
        if not ret: break

        # A. EVAPORATION: Slightly fade old positions
        heatmap_memory *= DECAY_RATE

        # B. AI SEGMENTATION: Find the person
        results = model(frame, verbose=False, conf=0.5, classes=[0])

        if results[0].masks is not None:
            # Create a 1/0 mask of the human silhouette
            masks = results[0].masks.data.cpu().numpy()
            combined_mask = np.any(masks, axis=0).astype(np.float32)
            
            # C. INJECT HEAT: Add value to the pixels you occupy
            heatmap_memory += (combined_mask * HEAT_GAIN)

        # D. CAP AND BLUR: This creates the smooth 'Glow' effect
        heatmap_memory = np.clip(heatmap_memory, 0, 255)
        
        # Apply blur to the heat memory to get that fuzzy "Presence Probability" look
        blurred_heat = cv2.GaussianBlur(heatmap_memory, (BLUR_SIZE, BLUR_SIZE), 0)
        
        # E. VISUALIZATION
        heat_norm = blurred_heat.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

        # F. BACKGROUND DIMMING (Desaturate and Darken)
        # We make the room darker so the glow pops out
        bg_dimmed = cv2.convertScaleAbs(frame, alpha=BG_BRIGHTNESS, beta=0)

        # G. BLENDING: Add the heatmap glow on top of the dimmed room
        # Only show heatmap where there is a value > 5
        mask = heat_norm > 5
        output = bg_dimmed.copy()
        
        # Weighted blend for the 'glowing' pixels
        output[mask] = cv2.addWeighted(bg_dimmed, 0.4, heatmap_color, 0.6, 0)[mask]

        # H. ADD THE PROBABILITY SCALE (Like in your image)
        cv2.putText(output, "Presence Probability: High (Red) / Low (Blue)", (20, h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Presence Probability Heatmap', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()