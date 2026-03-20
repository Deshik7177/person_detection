#!/usr/bin/env python
"""
Camera Diagnostic Tool
Tests all available cameras and displays information
"""

import cv2
import sys

def test_cameras():
    """Test all available camera indices"""
    print("=" * 60)
    print("🎥 CAMERA DIAGNOSTIC TOOL")
    print("=" * 60)
    
    found_any = False
    
    for camera_idx in range(10):
        print(f"\n[Testing Camera Index {camera_idx}]")
        cap = cv2.VideoCapture(camera_idx)
        
        if not cap.isOpened():
            print(f"  ✗ Not available")
            cap.release()
            continue
        
        found_any = True
        print(f"  ✓ Camera FOUND")
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"    Resolution: {width}x{height}")
        print(f"    FPS: {fps}")
        
        # Try to grab a frame
        ret, frame = cap.read()
        
        if ret and frame is not None:
            print(f"    Frame Grab: ✓ SUCCESS")
            print(f"    Frame Shape: {frame.shape}")
            print(f"    → Use camera_idx = {camera_idx} in main.py")
        else:
            print(f"    Frame Grab: ✗ FAILED (Camera may not have permission)")
        
        cap.release()
    
    print(f"\n" + "=" * 60)
    if found_any:
        print("✓ At least one camera was detected")
        print("\nUpdate main.py with the camera index that worked above:")
        print("  cap = cv2.VideoCapture(X)  # Replace X with the camera index")
    else:
        print("✗ NO CAMERAS DETECTED")
        print("\nTroubleshooting:")
        print("  1. Check if your camera is properly connected")
        print("  2. Check Device Manager for camera devices")
        print("  3. Ensure no other application is using the camera")
        print("  4. Try restarting your computer")
        print("  5. Check camera driver is installed")
    
    print("=" * 60)

if __name__ == "__main__":
    test_cameras()
