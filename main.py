import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, render_template, jsonify, Response
from flask_cors import CORS
import serial
import threading
import time

app = Flask(__name__)
CORS(app)

# Global variables
latest_frame = None
frame_lock = threading.Lock()
serial_conn = None

SERIAL_PORT = 'COM7'
BAUD_RATE = 115200

def init_serial():
    """Initialize serial connection to ESP32"""
    global serial_conn
    try:
        serial_conn = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"✓ Connected to ESP32 on {SERIAL_PORT} @ {BAUD_RATE} baud")
        return True
    except Exception as e:
        print(f"✗ Serial Connection Failed: {e}")
        serial_conn = None
        return False

def send_command(cmd):
    """Send command to ESP32"""
    try:
        if serial_conn and serial_conn.is_open:
            serial_conn.write((cmd + '\n').encode())
            print(f"→ Sent: {cmd}")
    except Exception as e:
        print(f"✗ Serial Error: {e}")

def frame_generator():
    """Generate frames for web streaming - optimized for low latency"""
    global latest_frame
    
    while True:
        with frame_lock:
            if latest_frame is not None:
                # Use MJPEG with lower quality for faster streaming
                ret, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 25])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
                       + frame_bytes + b'\r\n')
        # No sleep - stream as fast as possible

def video_processing():
    """Main video processing loop with low-latency heatmap"""
    global latest_frame
    
    import time as time_module
    
    # Wait for Flask to initialize
    time_module.sleep(1)
    
    model = YOLO('yolov8n-seg.pt')
    
    # Try multiple camera indices to find working camera
    cap = None
    camera_index = 0
    retry_count = 0
    max_retries = 3
    
    while cap is None and retry_count < max_retries:
        for idx in range(5):
            try:
                cap_test = cv2.VideoCapture(idx)
                cap_test.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Try to grab a frame to verify camera works
                ret, test_frame = cap_test.read()
                
                if ret and test_frame is not None and cap_test.get(cv2.CAP_PROP_FRAME_WIDTH) > 0:
                    cap = cap_test
                    camera_index = idx
                    print(f"✓ Camera found at index {idx} - FRAME GRAB SUCCESS")
                    break
                else:
                    cap_test.release()
                    time_module.sleep(0.5)
            except Exception as e:
                try:
                    cap_test.release()
                except:
                    pass
                time_module.sleep(0.5)
        
        if cap is None:
            retry_count += 1
            if retry_count < max_retries:
                print(f"⚠ Camera not found. Retry {retry_count}/{max_retries}...")
                time_module.sleep(2)
    
    if cap is None:
        print(f"✗ Camera not found after {max_retries} retries")
        print(f"  - Ensure camera is connected and not in use")
        print(f"  - Close any other camera applications")
        return
    
    # Optimize capture settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get frame dimensions
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Heat Accumulator
    heatmap_memory = np.zeros((h, w), dtype=np.float32)
    
    # Tuning parameters - settings from reference code
    DECAY_RATE = 0.75       # Faster decay for responsive heat
    HEAT_GAIN = 120.0       # High gain = Faster Red buildup
    
    frame_count = 0
    
    print("✓ Video Processing Started - Red-Point Intervention Detection")
    print(f"  Camera Index: {camera_index}")
    print(f"  Input resolution: {w}x{h}")
    print(f"  Mode: DECAY_RATE={DECAY_RATE}, HEAT_GAIN={HEAT_GAIN}")
    print(f"  YOLO inference: EVERY FRAME for accuracy")
    
    error_streak = 0
    max_errors = 5
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                error_streak += 1
                if error_streak > max_errors:
                    print("⚠ Camera connection lost - too many frame errors")
                    break
                time_module.sleep(0.1)
                continue
            
            error_streak = 0  # Reset on successful frame
            
            # 1. Decay the old heat
            heatmap_memory *= DECAY_RATE
            
            # 2. AI Segmentation - RUN EVERY FRAME for best detection
            try:
                results = model(frame, verbose=False, conf=0.5, classes=[0])
                
                if results[0].masks is not None:
                    masks = results[0].masks.data.cpu().numpy()
                    combined_mask = np.any(masks, axis=0).astype(np.float32)
                    
                    # 3. Inject Massive Heat
                    heatmap_memory += (combined_mask * HEAT_GAIN)
            except:
                pass  # Skip YOLO on error, continue with momentum
            
            # 4. CAP AT 255 (This is the Red Threshold)
            heatmap_memory = np.clip(heatmap_memory, 0, 255)
            
            # 5. Apply Glow - Medium blur for speed vs smoothness balance
            blurred_heat = cv2.GaussianBlur(heatmap_memory, (21, 21), 0)
            
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

            frame_count += 1
            
            with frame_lock:
                latest_frame = output

        except Exception as e:
            print(f"Processing error: {e}")
            error_streak += 1
            if error_streak > max_errors:
                break
            time_module.sleep(0.1)
    
    cap.release()
    print("✗ Video processing stopped")

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream heatmap video"""
    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/command/<cmd>')
def command(cmd):
    """Handle control commands"""
    commands = {
        'forward': 'F',
        'backward': 'B',
        'left': 'L',
        'right': 'R',
        'stop': 'S'
    }
    
    if cmd in commands:
        send_command(commands[cmd])
        return jsonify({'status': 'sent', 'command': cmd})
    return jsonify({'status': 'error', 'message': 'Unknown command'})

if __name__ == "__main__":
    init_serial()
    
    video_thread = threading.Thread(target=video_processing, daemon=True)
    video_thread.start()
    
    print("\n🚨 RESCUE PORTAL ACTIVE - Starting Web Server")
    print("📡 Open browser: http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)