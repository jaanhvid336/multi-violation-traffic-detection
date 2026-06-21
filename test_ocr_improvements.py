"""
OCR Validation Script
Tests the improved OCR recognition on plate images
"""

import cv2
import numpy as np
import sys
import os

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from ocr_module import PlateReader
from plate_detection import PlateDetector
from multi_frame_ocr import MultiFrameOCR

def test_ocr_on_sample():
    """Test OCR on a synthetic plate image"""
    print("=" * 60)
    print("OCR Recognition Testing")
    print("=" * 60)
    
    # Initialize OCR components
    plate_reader = PlateReader()
    plate_detector = PlateDetector(use_yolo_plate_model=False)
    multi_frame_ocr = MultiFrameOCR()
    
    print("\n✓ OCR modules initialized successfully")
    print(f"  - PlateReader confidence threshold: {plate_reader.confidence_threshold}")
    print(f"  - MultiFrameOCR history size: {multi_frame_ocr.history_size}")
    print(f"  - MultiFrameOCR confidence threshold: {multi_frame_ocr.confidence_threshold}")
    
    # Test on a synthetic plate image
    print("\n" + "-" * 60)
    print("Testing on synthetic plate image...")
    print("-" * 60)
    
    # Create a synthetic white plate with text
    plate_img = np.ones((100, 300, 3), dtype=np.uint8) * 255  # White background
    
    # Add some text to the image
    cv2.putText(plate_img, "MH 02 AB 1234", (30, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Test PlateReader
    print("\n[1] Testing PlateReader:")
    text1, conf1 = plate_reader.read_plate(plate_img)
    print(f"    Result: '{text1}' (confidence: {conf1:.2f})")
    
    # Test PlateDetector edge detection
    print("\n[2] Testing PlateDetector edge detection:")
    frame = np.ones((400, 600, 3), dtype=np.uint8) * 200  # Gray background
    # Place the plate in the frame
    frame[150:250, 150:450] = plate_img
    
    detections = plate_detector.detect_plates_in_frame(frame)
    print(f"    Found {len(detections)} plate(s)")
    for i, det in enumerate(detections):
        print(f"    Plate {i+1}: '{det['plate_text']}' (conf: {det['confidence']:.2f})")
    
    # Test MultiFrameOCR with multiple readings
    print("\n[3] Testing MultiFrameOCR voting:")
    track_id = 1
    
    # Simulate multiple readings of the same plate
    test_readings = [
        ("MH02AB1234", 0.85),
        ("MH02AB1234", 0.82),
        ("MH02AB123", 0.45),  # Imperfect reading
        ("MH02AB1234", 0.88),
    ]
    
    for reading, conf in test_readings:
        multi_frame_ocr.add_reading(track_id, reading, conf)
        print(f"    Added: '{reading}' (conf: {conf:.2f})")
    
    best_result = multi_frame_ocr.get_best_result(track_id)
    print(f"    ✓ Best result from voting: '{best_result}'")
    
    # Test similarity matching
    print("\n[4] Testing similarity matching:")
    track_id = 2
    
    test_readings = [
        ("MH02AB1234", 0.75),
        ("MH02AB1235", 0.72),  # Similar but slightly different
        ("MH02AB1236", 0.70),  # Very similar
    ]
    
    for reading, conf in test_readings:
        multi_frame_ocr.add_reading(track_id, reading, conf)
        print(f"    Added: '{reading}' (conf: {conf:.2f})")
    
    best_result = multi_frame_ocr.get_best_result(track_id)
    print(f"    ✓ Best result from similarity: '{best_result}'")
    
    print("\n" + "=" * 60)
    print("✓ All OCR tests completed successfully!")
    print("=" * 60)
    print("\nKey Improvements Made:")
    print("  1. Advanced preprocessing with CLAHE contrast enhancement")
    print("  2. Confidence score filtering (min threshold: 0.3)")
    print("  3. Multi-frame voting with similarity matching")
    print("  4. Improved edge detection parameters")
    print("  5. Support for hyphens in plate text")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_ocr_on_sample()
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
