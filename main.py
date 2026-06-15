import cv2
import time
import numpy as np
import os
from collections import deque
from datetime import datetime

from utils.emotion_detector import EmotionDetector
from utils.gaze_estimator import GazeEstimator
from utils.report_generator import ReportGenerator
from utils.audio_monitor import AudioMonitor
from utils.tab_monitor import TabMonitor
from brain_router import BrainRouter


class FocusAnalyzer:
    """Real-time focus analyzer with modern UI overlay."""
    
    # Color scheme (BGR format)
    COLORS = {
        'bg_dark': (40, 40, 40),
        'bg_panel': (50, 50, 50),
        'green': (0, 255, 100),
        'red': (0, 0, 255),
        'blue': (255, 180, 0),
        'yellow': (0, 220, 255),
        'white': (255, 255, 255),
        'gray': (150, 150, 150),
        'orange': (0, 140, 255),
    }

    def __init__(self):
        print("Initializing Focus Analyzer...")
        self.emotion_detector = EmotionDetector()
        self.gaze_estimator = GazeEstimator()
        self.report_gen = ReportGenerator()
        self.brain_router = BrainRouter()
        # Reduce audio chunk duration to reduce blocking wait time in get_noise_level
        self.audio_monitor = AudioMonitor(sample_rate=22050, chunk_duration=0.1)
        self.tab_monitor = TabMonitor()

        # Data storage
        self.data_buffer = deque(maxlen=3600)
        self.focus_history = deque(maxlen=100)  # For graph display
        
        # Metrics
        self.distraction_count = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.frames_processed = 0
        
        # State
        self.current_focus = 50
        self.current_emotion = 'neutral'
        self.current_gaze = True
        self.is_paused = False
        self.emotion_confidence = 0.5
        self.emotions_dict = {}
        self.current_noise_level = 0
        self.current_app = None
        self.rapid_switch_count = 0
        self.away_counted = False
        
        print("Focus Analyzer ready!")

    def compute_focus_score(self, emotion, gaze, emotions_dict=None):
        """Compute focus score based on emotion, gaze, audio, and tab switching."""
        # Handle 'away' state (no face detected)
        if emotion == 'away' or not gaze:
            return max(20, self.current_focus - 5)  # Gradual decrease when away
        
        score = 40  # Baseline when face detected
        
        # Gaze contribution (+35 points if looking at screen)
        if gaze:
            score += 35
        
        # Emotion contribution (+/- 25 points)
        emotion_scores = {
            'neutral': 15,
            'happy': 25,
            'surprise': 10,
            'sad': -10,
            'angry': -15,
            'fear': -10,
            'disgust': -15
        }
        score += emotion_scores.get(emotion, 0)
        
        # Use emotion confidence to weight the score
        if emotions_dict and emotion in emotions_dict:
            confidence = emotions_dict.get(emotion, 0.5)
            # Higher confidence = more extreme score
            if score > 50:
                score = 50 + int((score - 50) * (0.5 + confidence * 0.5))
            else:
                score = 50 - int((50 - score) * (0.5 + confidence * 0.5))

        # Audio environment factor
        audio_factor = self.audio_monitor.get_audio_quality_factor()
        score = int(score * audio_factor)
        
        # Tab switching factor
        switch_factor = self.tab_monitor.get_distraction_factor()
        score = int(score * switch_factor)
        
        return int(np.clip(score, 0, 100))

    def draw_focus_meter(self, frame, focus_score, x, y, width=200, height=25):
        """Draw an attractive focus meter bar."""
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.COLORS['bg_panel'], -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.COLORS['gray'], 1)
        
        # Calculate fill
        fill_width = int((focus_score / 100) * (width - 4))
        
        # Color based on score
        if focus_score >= 70:
            color = self.COLORS['green']
        elif focus_score >= 40:
            color = self.COLORS['yellow']
        else:
            color = self.COLORS['red']
        
        # Fill bar
        cv2.rectangle(frame, (x + 2, y + 2), (x + 2 + fill_width, y + height - 2), color, -1)
        
        # Text
        text = f"{focus_score}%"
        cv2.putText(frame, text, (x + width + 10, y + height - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['white'], 2)

    def draw_mini_graph(self, frame, data, x, y, width=200, height=60):
        """Draw a mini line graph of focus history."""
        if len(data) < 2:
            return
        
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.COLORS['bg_panel'], -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.COLORS['gray'], 1)
        
        # Draw grid lines
        for i in range(1, 4):
            gy = y + int(height * i / 4)
            cv2.line(frame, (x, gy), (x + width, gy), self.COLORS['gray'], 1)
        
        # Plot points
        points = []
        step = width / max(len(data) - 1, 1)
        for i, val in enumerate(data):
            px = int(x + i * step)
            py = int(y + height - (val / 100) * height)
            points.append((px, py))
        
        # Draw line
        for i in range(len(points) - 1):
            color = self.COLORS['green'] if data[i] >= 50 else self.COLORS['red']
            cv2.line(frame, points[i], points[i + 1], color, 2)

    def draw_ui(self, frame):
        """Draw the main UI overlay."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Semi-transparent panel on the right
        panel_width = 250
        cv2.rectangle(overlay, (w - panel_width, 0), (w, h), self.COLORS['bg_dark'], -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        
        # Panel content
        px = w - panel_width + 15
        
        # Title
        cv2.putText(frame, "FOCUS ANALYZER", (px, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['blue'], 2)
        
        # Divider
        cv2.line(frame, (px, 45), (w - 15, 45), self.COLORS['gray'], 1)
        
        # Focus Score
        cv2.putText(frame, "Focus Score", (px, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['gray'], 1)
        self.draw_focus_meter(frame, self.current_focus, px, 80, width=180)
        
        # Focus Graph
        cv2.putText(frame, "Focus Trend", (px, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['gray'], 1)
        self.draw_mini_graph(frame, list(self.focus_history), px, 140, width=210, height=50)
        
        # Status indicators
        cv2.putText(frame, "Status", (px, 215), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['gray'], 1)
        
        # Emotion with confidence
        emotion_colors = {
            'happy': self.COLORS['green'],
            'neutral': self.COLORS['white'],
            'surprise': self.COLORS['yellow'],
            'sad': self.COLORS['blue'],
            'angry': self.COLORS['red'],
            'fear': self.COLORS['orange'],
            'disgust': self.COLORS['red'],
            'away': self.COLORS['gray']
        }
        emotion_color = emotion_colors.get(self.current_emotion, self.COLORS['white'])
        emotion_text = f"{self.current_emotion.capitalize()} ({int(self.emotion_confidence*100)}%)"
        cv2.putText(frame, "Emotion:", (px, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['white'], 1)
        cv2.putText(frame, emotion_text, (px + 70, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, emotion_color, 2)
        
        # Gaze
        gaze_text = "On Screen" if self.current_gaze else "Away"
        gaze_color = self.COLORS['green'] if self.current_gaze else self.COLORS['red']
        cv2.putText(frame, "Gaze:", (px, 265), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['white'], 1)
        cv2.putText(frame, gaze_text, (px + 70, 265), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_color, 2)
        
        # Audio environment
        noise_text = f"Noise: {int(self.current_noise_level)}"
        if self.current_noise_level < 30:
            noise_color = self.COLORS['green']
        elif self.current_noise_level < 60:
            noise_color = self.COLORS['yellow']
        else:
            noise_color = self.COLORS['red']
        cv2.putText(frame, noise_text, (px, 290), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, noise_color, 1)
        
        # Tab switching indicator (left column)
        app_text = f"App: {self.current_app[:12]}" if self.current_app else "App: N/A"
        switches_per_min = self.tab_monitor.get_switch_frequency()
        if switches_per_min > 4:
            switch_color = self.COLORS['red']
        elif switches_per_min > 2:
            switch_color = self.COLORS['yellow']
        else:
            switch_color = self.COLORS['green']
        cv2.putText(frame, app_text, (px, 305), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['white'], 1)
        cv2.putText(frame, f"Switches: {self.rapid_switch_count}/min", (px, 325), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, switch_color, 1)
        
        # Session info
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        cv2.putText(frame, f"Time: {minutes:02d}:{seconds:02d}", (px, 350), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['white'], 1)
        cv2.putText(frame, f"Distractions: {self.distraction_count}", (px, 370), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['white'], 1)
        cv2.putText(frame, f"Samples: {len(self.data_buffer)}", (px, 390), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['white'], 1)
        # Controls
        cv2.line(frame, (px, h - 80), (w - 15, h - 80), self.COLORS['gray'], 1)
        cv2.putText(frame, "Controls", (px, h - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['gray'], 1)
        cv2.putText(frame, "Q - Quit & Report", (px, h - 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['white'], 1)
        cv2.putText(frame, "P - Pause/Resume", (px, h - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['white'], 1)
        
        # Paused indicator
        if self.is_paused:
            cv2.rectangle(frame, (w//2 - 80, h//2 - 30), (w//2 + 80, h//2 + 30), 
                         self.COLORS['bg_dark'], -1)
            cv2.rectangle(frame, (w//2 - 80, h//2 - 30), (w//2 + 80, h//2 + 30), 
                         self.COLORS['yellow'], 2)
            cv2.putText(frame, "PAUSED", (w//2 - 45, h//2 + 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLORS['yellow'], 2)
        
        # Low focus warning
        if self.current_focus < 40:
            text = "Refocus Needed!"
            font_scale = 1.0
            thickness = 3
            color = self.COLORS['red']
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x = max(10, w//2 - text_width//2)
            y = h - 50
            # Background rectangle
            cv2.rectangle(frame, (x - 10, y - text_height - 10), (min(w-10, x + text_width + 10), y + 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (x - 10, y - text_height - 10), (min(w-10, x + text_width + 10), y + 10), color, 2)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        elif self.current_focus > 70:
            text = "Great Focus!"
            font_scale = 1.0
            thickness = 3
            color = self.COLORS['green']
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x = max(10, w//2 - text_width//2)
            y = h - 50
            # Background rectangle
            cv2.rectangle(frame, (x - 10, y - text_height - 10), (min(w-10, x + text_width + 10), y + 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (x - 10, y - text_height - 10), (min(w-10, x + text_width + 10), y + 10), color, 2)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # Break reminder
        elapsed = time.time() - self.start_time
        if elapsed > 300:  # 5 minutes
            cv2.putText(frame, "Time for a 5-min break!", (w//2 - 150, h//2 + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLORS['yellow'], 2)
        
        # High distraction warning
        if self.distraction_count >= 8:
            cv2.putText(frame, "High Distractions! Session will end soon.", (w//2 - 200, h//2 + 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLORS['red'], 2)
        
        return frame

    def run(self):
        """Main application loop."""
        print("\nStarting camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera!")
            print("Make sure your webcam is connected and not in use.")
            return
        
        # Use lower resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 24)
        
        print("Camera started!")
        print("Focus tracking active. Press 'Q' to quit and generate report.\n")
        
        frame_count = 0
        last_gaze = True
        face_box = None
        target_focus = 50
        last_face_check = 0
        distraction_start_time = None
        last_sound_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame = cv2.flip(frame, 1)  # Mirror the frame
            frame_count += 1
            
            if not self.is_paused:
                # Get audio environment from non-blocking stream; returns latest smoothed value.
                noise_level, is_quiet = self.audio_monitor.get_noise_level()
                self.current_noise_level = noise_level

                # Monitor tab switching (still frequent)
                if frame_count % 15 == 0:
                    tab_info = self.tab_monitor.detect_tab_switch()
                    self.current_app = tab_info['current_app']
                    if tab_info['rapid']:
                        self.rapid_switch_count += 1

                # Detect emotion (every 4 frames)
                if frame_count % 4 == 0:
                    emotion, emotions_dict = self.emotion_detector.detect(frame)
                    self.current_emotion = emotion if emotion else 'neutral'
                    self.emotions_dict = emotions_dict if emotions_dict else {}
                else:
                    emotion = self.current_emotion
                    emotions_dict = self.emotions_dict

                # Store emotion confidence
                self.emotion_confidence = self.emotions_dict.get(self.current_emotion, 0.5)

                # Detect gaze (every 2 frames)
                if frame_count % 2 == 0:
                    gaze = self.gaze_estimator.is_looking_at_screen(frame)
                    self.current_gaze = gaze
                else:
                    gaze = self.current_gaze

                # Track away-duration distraction and play reminder every 5 seconds
                if not gaze:
                    if distraction_start_time is None:
                        distraction_start_time = time.time()
                        self.away_counted = False

                    elapsed_distraction = time.time() - distraction_start_time

                    if elapsed_distraction >= 2 and not self.away_counted:
                        self.distraction_count += 1
                        self.away_counted = True

                    if elapsed_distraction >= 5 and time.time() - last_sound_time >= 5:
                        os.system("say 'Please focus on your task and maintain concentration' &")
                        last_sound_time = time.time()
                else:
                    distraction_start_time = None
                    last_sound_time = 0
                    self.away_counted = False

                # Check for excessive distractions - cut session if too many
                if self.distraction_count >= 10:
                    print("Too many distractions detected! Ending session early to encourage better focus.")
                    break

                # Get face box less frequently to reduce workload
                if frame_count - last_face_check >= 10:
                    face_box = self.gaze_estimator.get_face_box(frame)
                    last_face_check = frame_count

                # Compute target focus score
                target_focus = self.compute_focus_score(
                    self.current_emotion, gaze, emotions_dict
                )
                
                # Smooth transition to target (reduces jitter)
                if self.current_focus < target_focus:
                    self.current_focus = min(self.current_focus + 2, target_focus)
                elif self.current_focus > target_focus:
                    self.current_focus = max(self.current_focus - 2, target_focus)
                
                # Update focus history for graph
                if frame_count % 20 == 0:
                    self.focus_history.append(self.current_focus)
                
                # Track distractions (gaze away)
                if last_gaze and not gaze:
                    self.distraction_count += 1
                last_gaze = gaze
                
                # Log data every second
                now = time.time()
                if now - self.last_log_time >= 1.0:
                    self.data_buffer.append({
                        'timestamp': now,
                        'emotion': self.current_emotion,
                        'gaze': gaze,
                        'focus_score': self.current_focus,
                        'noise_level': self.current_noise_level,
                        'current_app': self.current_app,
                        'rapid_switches': self.rapid_switch_count,
                        'switches_per_min': self.tab_monitor.get_switch_frequency()
                    })
                    self.last_log_time = now
                    self.frames_processed += 1
            
            # Draw face box only when looking at screen and face detected
            if face_box is not None and self.current_gaze and self.current_emotion != 'away':
                x, y, w, h = face_box
                # Color based on emotion
                emotion_box_colors = {
                    'happy': self.COLORS['green'],
                    'neutral': self.COLORS['blue'],
                    'surprise': self.COLORS['yellow'],
                    'sad': (255, 150, 0),  # Light blue
                    'angry': self.COLORS['red'],
                    'fear': self.COLORS['orange'],
                    'disgust': (128, 0, 128)  # Purple
                }
                box_color = emotion_box_colors.get(self.current_emotion, self.COLORS['white'])
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                # Label with emotion and focus
                label = f"{self.current_emotion.capitalize()} | Focus: {self.current_focus}%"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            elif not self.current_gaze or self.current_emotion == 'away':
                # Clear face box when not looking
                face_box = None
            
            # Draw UI
            frame = self.draw_ui(frame)
            
            # Show frame
            cv2.imshow('Focus Analyzer', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nStopping analysis...")
                break
            elif key == ord('p'):
                self.is_paused = not self.is_paused
                status = "Paused" if self.is_paused else "Resumed"
                print(status)

        cap.release()
        cv2.destroyAllWindows()
        # Stop non-blocking audio stream cleanly
        self.audio_monitor.stop()

        # Generate report
        print("\nGenerating report...")
        duration = (time.time() - self.start_time) / 60
        self.report_gen.generate(
            data=list(self.data_buffer),
            duration_min=duration,
            distraction_count=self.distraction_count
        )

        # Brain routing: provide post-session suggestions based on full session behavior.
        brain_results = self.brain_router.analyze_session(
            data_buffer=list(self.data_buffer),
            duration_min=duration,
            distraction_count=self.distraction_count
        )
        self.brain_router.analysis = brain_results.get('analysis', {})
        self.brain_router.suggestions = brain_results.get('suggestions', [])
        self.brain_router.print_suggestions()
        print("\nSession complete!")


if __name__ == "__main__":
    print("""
    ==================================================
             FOCUS ANALYZER v1.0                   
       Real-time Focus & Emotion Tracking          
    ==================================================
    """)
    
    analyzer = FocusAnalyzer()
    analyzer.run()