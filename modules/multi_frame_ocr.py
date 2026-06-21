from collections import Counter
import difflib

class MultiFrameOCR:
    def __init__(self, history_size=10):  # Increased history for better voting
        self.history_size = history_size
        self.plate_history = {} # track_id -> list of (text, prob, confidence)
        self.confidence_threshold = 0.25  # Accept readings with good confidence
        
    def add_reading(self, track_id, text, prob):
        """Add a plate reading with confidence score"""
        if not text or len(text) < 2:
            return
            
        if track_id not in self.plate_history:
            self.plate_history[track_id] = []
            
        self.plate_history[track_id].append((text, prob))
        
        # Keep only recent history
        if len(self.plate_history[track_id]) > self.history_size:
            self.plate_history[track_id].pop(0)
            
    def get_best_result(self, track_id):
        """Get the best plate reading using weighted voting"""
        if track_id not in self.plate_history or not self.plate_history[track_id]:
            return None
        
        readings = self.plate_history[track_id]
        
        # Filter by minimum confidence
        high_conf_readings = [(t, p) for t, p in readings if p >= self.confidence_threshold]
        
        if not high_conf_readings:
            # If no high confidence readings, use all
            high_conf_readings = readings
        
        if not high_conf_readings:
            return None
        
        # Try exact match first (majority voting)
        texts = [t[0] for t in high_conf_readings]
        counter = Counter(texts)
        most_common = counter.most_common(1)
        
        if most_common and most_common[0][1] >= 2:  # At least 2 matches
            return most_common[0][0]
        
        # If no exact matches, find similar readings using difflib
        if len(texts) >= 2:
            best_match = self._find_consensus(texts)
            if best_match:
                return best_match
        
        # Return highest confidence reading
        if readings:
            best_reading = max(readings, key=lambda x: x[1])
            return best_reading[0]
        
        return None
    
    def _find_consensus(self, texts):
        """Find consensus among similar text readings"""
        if not texts:
            return None
        
        # Group similar texts
        groups = {}
        for text in texts:
            matched = False
            for existing_text in groups:
                # Calculate similarity ratio
                ratio = difflib.SequenceMatcher(None, text, existing_text).ratio()
                if ratio > 0.7:  # 70% similarity
                    groups[existing_text].append(text)
                    matched = True
                    break
            if not matched:
                groups[text] = [text]
        
        # Return the group with most members
        if groups:
            best_group = max(groups.items(), key=lambda x: len(x[1]))
            return best_group[0]  # Return the key (representative text)
        
        return None
