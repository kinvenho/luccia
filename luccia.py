#!/usr/bin/env python3
"""
Luccia - Real-time Emotion-Driven Art Generator
===============================================

A Python application that captures facial expressions via webcam,
classifies emotions, and generates dynamic abstract art using a GAN
influenced by user-selected art moods.

Author: Luccia Development Team
"""

import cv2
import numpy as np
import pygame
import threading
import time
import sys
import os
from typing import Tuple, List, Optional, Dict
import platform

# Check if running in Pyodide environment
IS_PYODIDE = platform.system() == "Emscripten"

# Import modules based on availability
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not available. Using OpenCV face detection instead.")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Using simulated emotion detection.")

# Global configuration
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 30

# Emotion types
EMOTIONS = ['happy', 'sad', 'angry', 'calm']

# Art moods
ART_MOODS = {
    'impressionist': {'name': 'Impressionist', 'description': 'Soft brushstrokes, light colors'},
    'surrealist': {'name': 'Surrealist', 'description': 'Dreamlike, abstract forms'},
    'cubist': {'name': 'Cubist', 'description': 'Geometric shapes, fragmented forms'},
    'expressionist': {'name': 'Expressionist', 'description': 'Bold colors, emotional intensity'},
    'minimalist': {'name': 'Minimalist', 'description': 'Simple forms, limited palette'}
}

class EmotionDetector:
    """Real-time emotion detection using computer vision."""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.current_emotion = 'calm'
        self.emotion_confidence = 0.5
        self.face_detected = False
        self.face_landmarks = None
        
        if DLIB_AVAILABLE:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Simulated emotion classifier (replace with actual model)
        self.emotion_classifier = self._create_simulated_classifier()
    
    def _create_simulated_classifier(self):
        """Create a simulated emotion classifier for demonstration."""
        def classify_emotion(face_img):
            # Simulate emotion detection based on image characteristics
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Simple heuristics for emotion simulation
            if brightness > 120:
                return 'happy', 0.8
            elif brightness < 80:
                return 'sad', 0.7
            elif contrast > 50:
                return 'angry', 0.6
            else:
                return 'calm', 0.9
        
        return classify_emotion
    
    def detect_face(self, frame):
        """Detect faces in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return (x, y, w, h), gray[y:y+h, x:x+w]
        return None, None
    
    def get_landmarks(self, frame, face_rect):
        """Extract facial landmarks using dlib."""
        if not DLIB_AVAILABLE or face_rect is None:
            return None
        
        x, y, w, h = face_rect
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = self.predictor(frame, dlib_rect)
        return landmarks
    
    def detect_emotion(self, frame):
        """Detect emotion in the current frame."""
        face_rect, face_img = self.detect_face(frame)
        
        if face_rect is not None:
            self.face_detected = True
            emotion, confidence = self.emotion_classifier(frame)
            self.current_emotion = emotion
            self.emotion_confidence = confidence
            self.face_landmarks = self.get_landmarks(frame, face_rect)
        else:
            self.face_detected = False
            self.current_emotion = 'calm'
            self.emotion_confidence = 0.3
        
        return self.current_emotion, self.emotion_confidence

class ArtGAN:
    """Generative Adversarial Network for abstract art generation."""
    
    def __init__(self):
        self.latent_dim = 100
        self.image_size = 256
        self.current_art = None
        self.art_history = []
        
        # Emotion-to-art mappings
        self.emotion_mappings = {
            'happy': {
                'colors': [(255, 255, 0), (255, 165, 0), (255, 0, 0)],  # Yellow, orange, red
                'shapes': 'curves',
                'intensity': 0.8,
                'complexity': 0.7
            },
            'sad': {
                'colors': [(100, 149, 237), (128, 128, 128), (70, 130, 180)],  # Blue, gray
                'shapes': 'soft',
                'intensity': 0.3,
                'complexity': 0.4
            },
            'angry': {
                'colors': [(255, 0, 0), (139, 0, 0), (0, 0, 0)],  # Red, dark red, black
                'shapes': 'jagged',
                'intensity': 1.0,
                'complexity': 0.9
            },
            'calm': {
                'colors': [(173, 216, 230), (255, 182, 193), (144, 238, 144)],  # Pastels
                'shapes': 'flowing',
                'intensity': 0.5,
                'complexity': 0.6
            }
        }
    
    def generate_art(self, emotion: str, art_mood: str, confidence: float) -> np.ndarray:
        """Generate abstract art based on emotion and art mood."""
        mapping = self.emotion_mappings.get(emotion, self.emotion_mappings['calm'])
        
        # Create base canvas
        canvas = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Apply emotion-based color palette
        colors = mapping['colors']
        num_colors = len(colors)
        
        # Generate abstract patterns based on emotion and mood
        if mapping['shapes'] == 'curves':
            canvas = self._generate_curved_patterns(canvas, colors, mapping['intensity'])
        elif mapping['shapes'] == 'soft':
            canvas = self._generate_soft_gradients(canvas, colors, mapping['intensity'])
        elif mapping['shapes'] == 'jagged':
            canvas = self._generate_jagged_patterns(canvas, colors, mapping['intensity'])
        else:  # flowing
            canvas = self._generate_flowing_patterns(canvas, colors, mapping['intensity'])
        
        # Apply art mood modifications
        canvas = self._apply_art_mood(canvas, art_mood, confidence)
        
        self.current_art = canvas
        self.art_history.append(canvas.copy())
        
        # Keep only last 10 pieces for memory management
        if len(self.art_history) > 10:
            self.art_history.pop(0)
        
        return canvas
    
    def _generate_curved_patterns(self, canvas, colors, intensity):
        """Generate curved, organic patterns."""
        height, width = canvas.shape[:2]
        
        for i in range(int(50 * intensity)):
            color = colors[i % len(colors)]
            center_x = np.random.randint(0, width)
            center_y = np.random.randint(0, height)
            radius = np.random.randint(20, 100)
            
            # Draw curved lines
            for angle in range(0, 360, 5):
                rad = np.radians(angle)
                x = int(center_x + radius * np.cos(rad))
                y = int(center_y + radius * np.sin(rad))
                
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(canvas, (x, y), 2, color, -1)
        
        return canvas
    
    def _generate_soft_gradients(self, canvas, colors, intensity):
        """Generate soft, gradient-based patterns."""
        height, width = canvas.shape[:2]
        
        for i, color in enumerate(colors):
            # Create gradient from center
            center_x, center_y = width // 2, height // 2
            
            for y in range(height):
                for x in range(width):
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    alpha = max(0, 1 - distance / (width * 0.7))
                    alpha *= intensity
                    
                    if alpha > 0:
                        canvas[y, x] = [int(c * alpha) for c in color]
        
        return canvas
    
    def _generate_jagged_patterns(self, canvas, colors, intensity):
        """Generate sharp, jagged patterns."""
        height, width = canvas.shape[:2]
        
        for i in range(int(30 * intensity)):
            color = colors[i % len(colors)]
            
            # Create jagged lines
            points = []
            for j in range(5):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                points.append([x, y])
            
            if len(points) >= 2:
                points = np.array(points, dtype=np.int32)
                cv2.polylines(canvas, [points], False, color, 3)
        
        return canvas
    
    def _generate_flowing_patterns(self, canvas, colors, intensity):
        """Generate flowing, organic patterns."""
        height, width = canvas.shape[:2]
        
        for i in range(int(40 * intensity)):
            color = colors[i % len(colors)]
            
            # Create flowing curves
            center_x = np.random.randint(0, width)
            center_y = np.random.randint(0, height)
            
            for t in range(100):
                t_norm = t / 100.0
                x = int(center_x + 50 * np.sin(t_norm * 4 * np.pi))
                y = int(center_y + 30 * np.cos(t_norm * 3 * np.pi))
                
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(canvas, (x, y), 3, color, -1)
        
        return canvas
    
    def _apply_art_mood(self, canvas, art_mood, confidence):
        """Apply art mood modifications to the canvas."""
        if art_mood == 'impressionist':
            # Add soft brushstroke effect
            kernel = np.ones((3, 3), np.uint8)
            canvas = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, kernel)
        
        elif art_mood == 'surrealist':
            # Add dreamlike distortions
            height, width = canvas.shape[:2]
            for y in range(0, height, 10):
                for x in range(0, width, 10):
                    offset_x = int(5 * np.sin(y / 20))
                    if 0 <= x + offset_x < width:
                        canvas[y, x] = canvas[y, (x + offset_x) % width]
        
        elif art_mood == 'cubist':
            # Add geometric fragmentation
            height, width = canvas.shape[:2]
            block_size = 20
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    if np.random.random() < 0.3:
                        cv2.rectangle(canvas, (x, y), (x + block_size, y + block_size), 
                                    (255, 255, 255), -1)
        
        elif art_mood == 'expressionist':
            # Increase contrast and intensity
            canvas = cv2.convertScaleAbs(canvas, alpha=1.5, beta=30)
        
        elif art_mood == 'minimalist':
            # Simplify the composition
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            canvas = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        return canvas

class AudioManager:
    """Manages audio cues matching the art's mood."""
    
    def __init__(self):
        self.audio_enabled = False
        self.current_tone = 440  # A4 note
        
        try:
            import pygame.mixer
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.audio_enabled = True
        except:
            print("Warning: Audio not available")
    
    def play_emotion_tone(self, emotion: str, intensity: float):
        """Play audio tone matching the emotion."""
        if not self.audio_enabled:
            return
        
        # Map emotions to frequencies
        emotion_frequencies = {
            'happy': 523,  # C5
            'sad': 349,    # F4
            'angry': 659,  # E5
            'calm': 440    # A4
        }
        
        frequency = emotion_frequencies.get(emotion, 440)
        duration = int(100 * intensity)  # Duration in milliseconds
        
        # Generate simple sine wave
        sample_rate = 22050
        samples = int(sample_rate * duration / 1000)
        wave = np.sin(2 * np.pi * frequency * np.arange(samples) / sample_rate)
        wave = (wave * 32767).astype(np.int16)
        
        # Play the tone (simplified - in real implementation, use proper audio library)
        pass  # Placeholder for audio playback

class LucciaUI:
    """User interface for Luccia application."""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Luccia - Emotion-Driven Art Generator")
        
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        self.selected_mood = 'impressionist'
        self.show_camera = True
        self.show_controls = True
        
        # UI colors
        self.colors = {
            'background': (20, 20, 20),
            'text': (255, 255, 255),
            'highlight': (100, 150, 255),
            'button': (60, 60, 60),
            'button_hover': (80, 80, 80)
        }
    
    def draw_art(self, art_canvas: np.ndarray):
        """Draw the generated art on the screen."""
        if art_canvas is None:
            return
        
        # Convert numpy array to pygame surface
        art_surface = pygame.surfarray.make_surface(art_canvas.swapaxes(0, 1))
        
        # Scale to fit the art display area
        art_rect = pygame.Rect(400, 50, 700, 500)
        scaled_art = pygame.transform.scale(art_surface, (art_rect.width, art_rect.height))
        
        self.screen.blit(scaled_art, art_rect)
    
    def draw_camera_feed(self, camera_surface):
        """Draw the camera feed on the screen."""
        if camera_surface is None:
            return
        
        camera_rect = pygame.Rect(50, 50, 300, 225)
        scaled_camera = pygame.transform.scale(camera_surface, (camera_rect.width, camera_rect.height))
        self.screen.blit(scaled_camera, camera_rect)
    
    def draw_controls(self, emotion: str, confidence: float, art_mood: str):
        """Draw UI controls and information."""
        # Background for controls
        controls_rect = pygame.Rect(50, 300, 300, 400)
        pygame.draw.rect(self.screen, self.colors['button'], controls_rect)
        
        # Title
        title = self.font.render("Luccia Controls", True, self.colors['text'])
        self.screen.blit(title, (60, 320))
        
        # Emotion display
        emotion_text = f"Emotion: {emotion.title()}"
        emotion_surface = self.small_font.render(emotion_text, True, self.colors['text'])
        self.screen.blit(emotion_surface, (60, 360))
        
        # Confidence bar
        conf_text = f"Confidence: {confidence:.2f}"
        conf_surface = self.small_font.render(conf_text, True, self.colors['text'])
        self.screen.blit(conf_surface, (60, 380))
        
        # Confidence bar visualization
        bar_rect = pygame.Rect(60, 400, 200, 20)
        pygame.draw.rect(self.screen, (100, 100, 100), bar_rect)
        fill_rect = pygame.Rect(60, 400, int(200 * confidence), 20)
        pygame.draw.rect(self.screen, self.colors['highlight'], fill_rect)
        
        # Art mood selection
        mood_text = f"Art Mood: {ART_MOODS[art_mood]['name']}"
        mood_surface = self.small_font.render(mood_text, True, self.colors['text'])
        self.screen.blit(mood_surface, (60, 440))
        
        # Instructions
        instructions = [
            "Press 1-5: Change art mood",
            "C: Toggle camera view",
            "S: Save current art",
            "A: Toggle audio",
            "ESC: Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            inst_surface = self.small_font.render(instruction, True, self.colors['text'])
            self.screen.blit(inst_surface, (60, 480 + i * 25))
    
    def handle_events(self):
        """Handle pygame events and return user actions."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 'quit'
                elif event.key == pygame.K_c:
                    self.show_camera = not self.show_camera
                elif event.key == pygame.K_s:
                    return 'save_art'
                elif event.key == pygame.K_a:
                    return 'toggle_audio'
                elif event.key == pygame.K_1:
                    return 'mood_impressionist'
                elif event.key == pygame.K_2:
                    return 'mood_surrealist'
                elif event.key == pygame.K_3:
                    return 'mood_cubist'
                elif event.key == pygame.K_4:
                    return 'mood_expressionist'
                elif event.key == pygame.K_5:
                    return 'mood_minimalist'
        
        return None

class Luccia:
    """Main application class for Luccia."""
    
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.art_gan = ArtGAN()
        self.audio_manager = AudioManager()
        self.ui = LucciaUI()
        
        self.camera = None
        self.running = False
        self.current_emotion = 'calm'
        self.emotion_confidence = 0.5
        self.current_art_mood = 'impressionist'
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
    
    def initialize_camera(self):
        """Initialize the webcam."""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("Warning: Could not open camera. Using simulated camera feed.")
                self.camera = None
            else:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        except Exception as e:
            print(f"Warning: Camera initialization failed: {e}")
            self.camera = None
    
    def get_camera_frame(self):
        """Get a frame from the camera or generate a simulated one."""
        if self.camera is not None and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                return frame
        
        # Generate simulated camera feed
        frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera Not Available", (50, CAMERA_HEIGHT//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame
    
    def save_art(self):
        """Save the current art as an image."""
        if self.art_gan.current_art is not None:
            timestamp = int(time.time())
            filename = f"luccia_art_{timestamp}.png"
            
            if not IS_PYODIDE:
                cv2.imwrite(filename, self.art_gan.current_art)
                print(f"Art saved as {filename}")
            else:
                print("File saving not available in Pyodide environment")
    
    def update_fps(self):
        """Update FPS calculation."""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def run(self):
        """Main application loop."""
        print("Starting Luccia - Emotion-Driven Art Generator")
        print("Controls:")
        print("  1-5: Change art mood")
        print("  C: Toggle camera view")
        print("  S: Save current art")
        print("  A: Toggle audio")
        print("  ESC: Quit")
        
        self.initialize_camera()
        self.running = True
        
        clock = pygame.time.Clock()
        
        while self.running:
            # Handle events
            action = self.ui.handle_events()
            
            if action == 'quit':
                self.running = False
            elif action == 'save_art':
                self.save_art()
            elif action == 'toggle_audio':
                self.audio_manager.audio_enabled = not self.audio_manager.audio_enabled
            elif action and action.startswith('mood_'):
                self.current_art_mood = action.replace('mood_', '')
            
            # Get camera frame
            camera_frame = self.get_camera_frame()
            
            # Detect emotion
            emotion, confidence = self.emotion_detector.detect_emotion(camera_frame)
            self.current_emotion = emotion
            self.emotion_confidence = confidence
            
            # Generate art
            art_canvas = self.art_gan.generate_art(emotion, self.current_art_mood, confidence)
            
            # Play audio cue
            if self.audio_manager.audio_enabled:
                self.audio_manager.play_emotion_tone(emotion, confidence)
            
            # Update display
            self.ui.screen.fill(self.ui.colors['background'])
            
            # Draw camera feed
            if self.ui.show_camera:
                camera_surface = pygame.surfarray.make_surface(
                    cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
                )
                self.ui.draw_camera_feed(camera_surface)
            
            # Draw art
            self.ui.draw_art(art_canvas)
            
            # Draw controls
            self.ui.draw_controls(emotion, confidence, self.current_art_mood)
            
            # Draw FPS
            fps_text = self.ui.small_font.render(f"FPS: {self.fps:.1f}", True, self.ui.colors['text'])
            self.ui.screen.blit(fps_text, (WINDOW_WIDTH - 100, 20))
            
            pygame.display.flip()
            
            # Update FPS
            self.update_fps()
            
            # Cap frame rate
            clock.tick(FPS)
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.camera is not None:
            self.camera.release()
        pygame.quit()
        print("Luccia closed.")

def main():
    """Main entry point for the application."""
    try:
        app = Luccia()
        app.run()
    except KeyboardInterrupt:
        print("\nLuccia interrupted by user.")
    except Exception as e:
        print(f"Error running Luccia: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
