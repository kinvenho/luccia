"""
Configuration file for Luccia - Emotion-Driven Art Generator
Contains all customizable settings for the application.
"""

import os
from typing import Dict, Any

# Display Settings
DISPLAY_CONFIG = {
    'width': 1200,
    'height': 800,
    'fps': 30,
    'fullscreen': False,
    'vsync': True,
    'title': 'Luccia - Emotion-Driven Art Generator'
}

# Camera Settings
CAMERA_CONFIG = {
    'device_id': 0,  # Default camera
    'width': 640,
    'height': 480,
    'fps': 30,
    'flip_horizontal': True,  # Mirror the camera feed
    'brightness': 0,
    'contrast': 0,
    'saturation': 0
}

# Emotion Detection Settings
EMOTION_CONFIG = {
    'detection_interval': 5,  # Process every N frames
    'confidence_threshold': 0.3,
    'face_scale_factor': 1.1,
    'min_neighbors': 5,
    'min_face_size': (30, 30),
    'model_type': 'simulated',  # 'simulated', 'tensorflow', 'pytorch'
    'use_landmarks': False,  # Whether to use dlib facial landmarks
    'emotion_smoothing': 0.7  # Smoothing factor for emotion changes
}

# Art Generation Settings
ART_CONFIG = {
    'canvas_width': 800,
    'canvas_height': 600,
    'update_rate': 10,  # Update art every N frames
    'complexity_factor': 1.0,
    'color_intensity': 1.0,
    'pattern_density': 0.5,
    'animation_speed': 1.0,
    'blend_mode': 'alpha',  # 'alpha', 'add', 'multiply'
    'max_shapes': 50,
    'min_shape_size': 10,
    'max_shape_size': 100
}

# Audio Settings
AUDIO_CONFIG = {
    'enabled': True,
    'sample_rate': 44100,
    'volume': 0.3,
    'fade_duration': 0.5,
    'frequency_range': {
        'happy': (440, 880),    # A4 to A5
        'sad': (220, 440),      # A3 to A4
        'angry': (880, 1760),   # A5 to A6
        'calm': (330, 660)      # E4 to E5
    },
    'waveform': 'sine'  # 'sine', 'square', 'triangle', 'sawtooth'
}

# UI Settings
UI_CONFIG = {
    'font_size': 16,
    'font_name': 'Arial',
    'colors': {
        'background': (20, 20, 20),
        'text': (255, 255, 255),
        'highlight': (100, 150, 255),
        'warning': (255, 100, 100),
        'success': (100, 255, 100)
    },
    'show_fps': True,
    'show_controls': True,
    'show_camera': True,
    'camera_position': (10, 10),
    'camera_size': (200, 150),
    'control_panel_width': 250
}

# Performance Settings
PERFORMANCE_CONFIG = {
    'max_fps': 60,
    'target_fps': 30,
    'enable_vsync': True,
    'multithreading': False,
    'gpu_acceleration': False,
    'memory_limit_mb': 512
}

# File Settings
FILE_CONFIG = {
    'save_directory': 'saved_art',
    'log_directory': 'logs',
    'auto_save': False,
    'save_interval': 30,  # Save every N seconds
    'image_format': 'png',
    'max_saved_files': 100
}

# Emotion-to-Art Mappings
EMOTION_ART_MAPPINGS = {
    'happy': {
        'colors': [(255, 255, 0), (255, 165, 0), (255, 255, 255)],  # Yellow, Orange, White
        'patterns': 'curved',
        'intensity': 0.8,
        'complexity': 0.6,
        'speed': 1.2
    },
    'sad': {
        'colors': [(100, 149, 237), (70, 130, 180), (25, 25, 112)],  # Blue tones
        'patterns': 'soft',
        'intensity': 0.4,
        'complexity': 0.3,
        'speed': 0.6
    },
    'angry': {
        'colors': [(255, 0, 0), (139, 0, 0), (255, 69, 0)],  # Red tones
        'patterns': 'jagged',
        'intensity': 1.0,
        'complexity': 0.9,
        'speed': 1.5
    },
    'calm': {
        'colors': [(144, 238, 144), (173, 216, 230), (221, 160, 221)],  # Pastels
        'patterns': 'flowing',
        'intensity': 0.5,
        'complexity': 0.4,
        'speed': 0.8
    }
}

# Art Mood Settings
ART_MOODS = {
    'impressionist': {
        'name': 'Impressionist',
        'description': 'Soft, flowing brushstrokes with natural colors',
        'color_palette': [(255, 223, 186), (255, 218, 185), (255, 192, 203)],
        'pattern_style': 'flowing',
        'intensity_modifier': 0.8,
        'complexity_modifier': 0.7
    },
    'surrealist': {
        'name': 'Surrealist',
        'description': 'Dreamlike, abstract forms with unexpected combinations',
        'color_palette': [(138, 43, 226), (255, 20, 147), (0, 255, 255)],
        'pattern_style': 'abstract',
        'intensity_modifier': 1.2,
        'complexity_modifier': 1.1
    },
    'cubist': {
        'name': 'Cubist',
        'description': 'Geometric shapes and fragmented forms',
        'color_palette': [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        'pattern_style': 'geometric',
        'intensity_modifier': 1.0,
        'complexity_modifier': 1.3
    },
    'minimalist': {
        'name': 'Minimalist',
        'description': 'Simple, clean lines with limited colors',
        'color_palette': [(255, 255, 255), (0, 0, 0), (128, 128, 128)],
        'pattern_style': 'simple',
        'intensity_modifier': 0.5,
        'complexity_modifier': 0.3
    },
    'expressionist': {
        'name': 'Expressionist',
        'description': 'Bold, emotional colors with dynamic brushstrokes',
        'color_palette': [(255, 0, 0), (255, 255, 0), (0, 0, 255)],
        'pattern_style': 'dynamic',
        'intensity_modifier': 1.4,
        'complexity_modifier': 1.0
    }
}

# Debug Settings
DEBUG_CONFIG = {
    'enabled': False,
    'log_level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'show_debug_info': False,
    'save_debug_frames': False,
    'performance_monitoring': False
}

# Pyodide Compatibility Settings
PYODIDE_CONFIG = {
    'check_platform': True,
    'disable_file_io': False,
    'use_async_loop': True,
    'memory_limit': 100 * 1024 * 1024  # 100MB
}

def get_config() -> Dict[str, Any]:
    """Get the complete configuration dictionary."""
    return {
        'display': DISPLAY_CONFIG,
        'camera': CAMERA_CONFIG,
        'emotion': EMOTION_CONFIG,
        'art': ART_CONFIG,
        'audio': AUDIO_CONFIG,
        'ui': UI_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'file': FILE_CONFIG,
        'emotion_mappings': EMOTION_ART_MAPPINGS,
        'art_moods': ART_MOODS,
        'debug': DEBUG_CONFIG,
        'pyodide': PYODIDE_CONFIG
    }

def update_config(section: str, key: str, value: Any) -> None:
    """Update a specific configuration value."""
    config = get_config()
    if section in config and key in config[section]:
        config[section][key] = value
    else:
        raise ValueError(f"Invalid section '{section}' or key '{key}'")

def validate_config() -> bool:
    """Validate the configuration settings."""
    try:
        config = get_config()
        
        # Check display settings
        if config['display']['width'] <= 0 or config['display']['height'] <= 0:
            return False
        
        # Check camera settings
        if config['camera']['device_id'] < 0:
            return False
        
        # Check emotion settings
        if not 0 <= config['emotion']['confidence_threshold'] <= 1:
            return False
        
        # Check art settings
        if config['art']['canvas_width'] <= 0 or config['art']['canvas_height'] <= 0:
            return False
        
        return True
    except Exception:
        return False

def create_directories() -> None:
    """Create necessary directories for the application."""
    config = get_config()
    
    # Create save directory
    save_dir = config['file']['save_directory']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create log directory
    log_dir = config['file']['log_directory']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

if __name__ == "__main__":
    # Test configuration
    print("Testing Luccia configuration...")
    
    if validate_config():
        print("✓ Configuration is valid")
        create_directories()
        print("✓ Directories created")
        
        config = get_config()
        print(f"Display: {config['display']['width']}x{config['display']['height']}")
        print(f"Camera: Device {config['camera']['device_id']}")
        print(f"Art Moods: {len(config['art_moods'])} available")
        print(f"Audio: {'Enabled' if config['audio']['enabled'] else 'Disabled'}")
    else:
        print("✗ Configuration validation failed")
