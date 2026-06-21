import easyocr
import cv2
import numpy as np

class PlateReader:
    def __init__(self):
        # Initialize EasyOCR reader for English
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.confidence_threshold = 0.3  # Lower threshold for clearer plates
        
    def _preprocess_plate(self, plate_crop):
        """
        Advanced preprocessing for plate recognition.
        """
        if plate_crop is None or plate_crop.size == 0:
            return None
            
        # Convert to LAB color space for better contrast enhancement
        try:
            lab = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            preprocessed = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        except:
            preprocessed = plate_crop
        
        # Convert to grayscale
        gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        return filtered
        
    def read_plate(self, plate_crop):
        """
        Reads text from a cropped image of a number plate with improved preprocessing.
        """
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0
        
        # Preprocess the plate image
        processed = self._preprocess_plate(plate_crop)
        if processed is None:
            return "", 0.0
        
        # Run OCR with detail=1 to get confidence scores
        results = self.reader.readtext(processed, detail=1)
        
        if not results:
            return "", 0.0
        
        # Collect all valid readings
        readings = []
        for (bbox, text, prob) in results:
            if prob >= self.confidence_threshold:
                # Clean text but preserve hyphens and numbers
                cleaned_text = ''.join(c for c in text if c.isalnum() or c in '-')
                if cleaned_text:
                    readings.append((cleaned_text.upper(), prob))
        
        if not readings:
            return "", 0.0
        
        # Combine all readings and find best match
        combined_text = ''.join([r[0] for r in readings])
        avg_prob = np.mean([r[1] for r in readings])
        
        return combined_text, avg_prob
