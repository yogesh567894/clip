import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

class SubjectTracker:
    """
    Advanced subject detection and tracking for intelligent video reframing.
    Uses a combination of face detection, object detection, and motion tracking
    to identify and track the main subject of interest in a video.
    """
    
    def __init__(self):
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize object tracking
        self.tracker = None
        self.tracker_type = "KCF"  # Options: BOOSTING, MIL, KCF, TLD, MEDIANFLOW, MOSSE, CSRT
        self.tracking_active = False
        
        # Subject state for tracking
        self.last_subject_box = None
        self.tracking_confidence = 0.0
        self.frames_since_detection = 0
        self.detection_frequency = 30  # Re-detect every N frames
        
        # Initialize subject history for smoothing
        self.subject_history = []
        self.history_length = 10
    
    def _create_tracker(self):
        """Create and return OpenCV tracker based on tracker_type"""
        if self.tracker_type == 'BOOSTING':
            return cv2.legacy.TrackerBoosting_create()
        elif self.tracker_type == 'MIL':
            return cv2.legacy.TrackerMIL_create()
        elif self.tracker_type == 'KCF':
            return cv2.legacy.TrackerKCF_create()
        elif self.tracker_type == 'TLD':
            return cv2.legacy.TrackerTLD_create()
        elif self.tracker_type == 'MEDIANFLOW':
            return cv2.legacy.TrackerMedianFlow_create()
        elif self.tracker_type == 'MOSSE':
            return cv2.legacy.TrackerMOSSE_create()
        elif self.tracker_type == 'CSRT':
            return cv2.legacy.TrackerCSRT_create()
        else:
            return cv2.legacy.TrackerKCF_create()
    
    def detect_subject(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detect the primary subject in a video frame.
        Returns (x, y, width, height) coordinates of the subject
        """
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # First try face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        subject_box = None
        detection_confidence = 0.0
        
        if len(faces) > 0:
            # Use the largest face as our primary subject
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Expand the face area to include more context
            expansion_factor = 1.5
            expanded_w = int(w * expansion_factor)
            expanded_h = int(h * expansion_factor)
            expanded_x = max(0, x - (expanded_w - w) // 2)
            expanded_y = max(0, y - (expanded_h - h) // 2)
            
            # Ensure the box stays within the frame
            if expanded_x + expanded_w > width:
                expanded_w = width - expanded_x
            if expanded_y + expanded_h > height:
                expanded_h = height - expanded_y
            
            subject_box = (expanded_x, expanded_y, expanded_w, expanded_h)
            detection_confidence = 0.8  # High confidence for face detection
        
        # If no faces detected, try using motion detection or rule of thirds
        if subject_box is None:
            # For now, use simple rule of thirds placement
            center_x, center_y = width // 2, height // 2
            roi_width, roi_height = int(width * 0.6), int(height * 0.6)
            
            subject_box = (
                center_x - roi_width // 2,
                center_y - roi_height // 2,
                roi_width,
                roi_height
            )
            detection_confidence = 0.3  # Low confidence for fallback method
        
        return subject_box, detection_confidence
    
    def initialize_tracking(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Initialize subject tracking with a new detection
        """
        # Reset tracking state
        self.tracking_active = False
        self.frames_since_detection = 0
        
        # Detect subject
        subject_box, confidence = self.detect_subject(frame)
        self.tracking_confidence = confidence
        
        # Initialize tracker with the detected subject
        self.tracker = self._create_tracker()
        x, y, w, h = subject_box
        bbox = (x, y, w, h)
        success = self.tracker.init(frame, bbox)
        
        if success:
            self.tracking_active = True
            self.last_subject_box = subject_box
            
            # Initialize history with current position
            self.subject_history = [subject_box] * self.history_length
        else:
            print("Failed to initialize tracker")
        
        return subject_box
    
    def track_subject(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Track the subject across frames.
        Returns (x, y, width, height) coordinates of the subject
        """
        # Increment counter
        self.frames_since_detection += 1
        
        # Re-detect subject periodically for better accuracy
        if (self.frames_since_detection > self.detection_frequency or 
            not self.tracking_active or 
            self.tracking_confidence < 0.4):
            return self.initialize_tracking(frame)
        
        # Update tracker
        if self.tracking_active:
            success, bbox = self.tracker.update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in bbox]
                self.last_subject_box = (x, y, w, h)
                
                # Add to history and remove oldest entry
                self.subject_history.append(self.last_subject_box)
                if len(self.subject_history) > self.history_length:
                    self.subject_history.pop(0)
                
                # Apply smoothing
                return self._smooth_subject_box()
        
        # If tracking fails, return last known position
        return self.last_subject_box if self.last_subject_box else self.initialize_tracking(frame)
    
    def _smooth_subject_box(self) -> Tuple[int, int, int, int]:
        """
        Smooth the subject box coordinates using recent history
        to prevent jittery camera movement
        """
        if not self.subject_history:
            return self.last_subject_box
        
        # Apply weighted average - recent positions matter more
        weights = list(range(1, len(self.subject_history) + 1))
        total_weight = sum(weights)
        
        # Calculate weighted average for each coordinate
        avg_x = sum(box[0] * w for box, w in zip(self.subject_history, weights)) / total_weight
        avg_y = sum(box[1] * w for box, w in zip(self.subject_history, weights)) / total_weight
        avg_w = sum(box[2] * w for box, w in zip(self.subject_history, weights)) / total_weight
        avg_h = sum(box[3] * w for box, w in zip(self.subject_history, weights)) / total_weight
        
        # Return smoothed values as integers
        return (int(avg_x), int(avg_y), int(avg_w), int(avg_h))
    
    def get_crop_window(self, 
                        frame: np.ndarray, 
                        target_aspect_ratio: float,
                        padding_percent: float = 0.1) -> Tuple[int, int, int, int]:
        """
        Calculate the optimal crop window based on subject position and target aspect ratio
        Returns (x, y, width, height) of the crop window
        """
        height, width = frame.shape[:2]
        source_aspect_ratio = width / height
        
        # Get the current subject position
        subject_x, subject_y, subject_w, subject_h = self.track_subject(frame)
        subject_center_x = subject_x + subject_w // 2
        subject_center_y = subject_y + subject_h // 2
        
        # Calculate crop dimensions based on target aspect ratio
        if target_aspect_ratio > source_aspect_ratio:
            # Target is wider than source - constrain by height
            crop_height = height
            crop_width = int(height * target_aspect_ratio)
        else:
            # Target is taller than source - constrain by width
            crop_width = width
            crop_height = int(width / target_aspect_ratio)
        
        # Apply padding
        padding_w = int(crop_width * padding_percent)
        padding_h = int(crop_height * padding_percent)
        
        # Adjust crop size with padding
        crop_width -= 2 * padding_w
        crop_height -= 2 * padding_h
        
        # Calculate crop position centered on subject
        crop_x = subject_center_x - crop_width // 2
        crop_y = subject_center_y - crop_height // 2
        
        # Ensure crop window is within frame boundaries
        if crop_x < 0:
            crop_x = 0
        elif crop_x + crop_width > width:
            crop_x = width - crop_width
            
        if crop_y < 0:
            crop_y = 0
        elif crop_y + crop_height > height:
            crop_y = height - crop_height
        
        return crop_x, crop_y, crop_width, crop_height
    
    def draw_debug_overlay(self, frame: np.ndarray, crop_box: Tuple[int, int, int, int] = None) -> np.ndarray:
        """
        Draw debug overlays showing subject detection and tracking information
        """
        debug_frame = frame.copy()
        
        # Draw subject box if available
        if self.last_subject_box:
            x, y, w, h = self.last_subject_box
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Subject ({self.tracking_confidence:.2f})", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw crop box if available
        if crop_box:
            x, y, w, h = crop_box
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(debug_frame, "Crop Window", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return debug_frame 