# Luccia - Real-time Emotion-Driven Art Generator

Luccia is a Python-based, real-time emotion-driven art generator that captures facial expressions via webcam, classifies emotions, and generates dynamic abstract art using a GAN influenced by user-selected art moods.

## Features

- **Real-time Emotion Detection**: Captures facial expressions and classifies emotions (happy, sad, angry, calm)
- **Dynamic Art Generation**: Uses a lightweight GAN to generate abstract art based on detected emotions
- **Art Mood Selection**: Choose from different artistic styles (Impressionist, Surrealist, Cubist, Expressionist, Minimalist)
- **Real-time Rendering**: Live art generation and display using Pygame
- **Audio Integration**: Optional audio cues matching the art's mood
- **Art Saving**: Save generated artwork as PNG images
- **Pyodide Compatibility**: Designed to work in browser environments

## Emotion-to-Art Mappings

| Emotion | Colors | Shapes | Intensity |
|---------|--------|--------|-----------|
| Happy | Bright yellows, oranges, reds | Curved, organic patterns | High |
| Sad | Cool blues, grays | Soft gradients | Low |
| Angry | Sharp reds, dark tones | Jagged, sharp patterns | Very High |
| Calm | Pastels, soft tones | Flowing, smooth patterns | Medium |

## Art Moods

- **Impressionist**: Soft brushstrokes, light colors
- **Surrealist**: Dreamlike, abstract forms with distortions
- **Cubist**: Geometric shapes, fragmented forms
- **Expressionist**: Bold colors, high contrast, emotional intensity
- **Minimalist**: Simple forms, limited palette

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (optional - will use simulated feed if unavailable)
- Audio system (optional)

### Setup

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd luccia
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Install dlib for advanced facial landmark detection**:
   ```bash
   # On Windows:
   pip install dlib
   
   # On macOS:
   brew install cmake
   pip install dlib
   
   # On Linux:
   sudo apt-get install cmake
   pip install dlib
   ```

4. **Optional: Download facial landmark model** (for dlib):
   ```bash
   # Download shape_predictor_68_face_landmarks.dat from:
   # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   # Extract and place in the project directory
   ```

## Usage

### Running the Application

```bash
python luccia.py
```

### Controls

| Key | Action |
|-----|--------|
| `1-5` | Change art mood (1=Impressionist, 2=Surrealist, 3=Cubist, 4=Expressionist, 5=Minimalist) |
| `C` | Toggle camera view |
| `S` | Save current art as PNG |
| `A` | Toggle audio cues |
| `ESC` | Quit application |

### Interface

The application window is divided into three main areas:

1. **Camera Feed** (top-left): Shows the webcam feed with face detection
2. **Art Display** (center-right): Shows the generated abstract art
3. **Controls Panel** (bottom-left): Displays current emotion, confidence, and controls

## Technical Details

### Architecture

- **EmotionDetector**: Handles facial detection and emotion classification
- **ArtGAN**: Generates abstract art based on emotions and art moods
- **AudioManager**: Manages audio cues matching the art's mood
- **LucciaUI**: Pygame-based user interface
- **Luccia**: Main application coordinator

### Performance Optimization

- Real-time processing optimized for modest hardware
- Efficient memory management (keeps only last 10 art pieces)
- Configurable FPS (default: 30 FPS)
- Graceful degradation when dependencies are unavailable

### Compatibility

- **Desktop**: Full functionality with webcam and audio
- **Pyodide/Browser**: Limited file I/O, simulated camera feed
- **Headless**: Runs without display (art generation only)

## Troubleshooting

### Common Issues

1. **Camera not working**:
   - Application will use simulated camera feed
   - Check webcam permissions
   - Ensure no other application is using the camera

2. **dlib installation issues**:
   - Application will fall back to OpenCV face detection
   - Install Visual Studio Build Tools (Windows)
   - Install cmake (macOS/Linux)

3. **Audio not working**:
   - Audio is optional and will be disabled if unavailable
   - Check system audio settings

4. **Performance issues**:
   - Reduce FPS in the code (change `FPS = 30` to lower value)
   - Disable camera view with 'C' key
   - Close other applications

### Error Messages

- `"Warning: dlib not available"`: Using OpenCV face detection
- `"Warning: TensorFlow not available"`: Using simulated emotion detection
- `"Warning: Camera not available"`: Using simulated camera feed
- `"Warning: Audio not available"`: Audio features disabled

## Development

### Adding New Art Moods

1. Add mood to `ART_MOODS` dictionary in `luccia.py`
2. Implement mood-specific modifications in `_apply_art_mood()` method
3. Add keyboard shortcut in `handle_events()` method

### Adding New Emotions

1. Add emotion to `EMOTIONS` list
2. Add emotion mapping to `emotion_mappings` in `ArtGAN` class
3. Update emotion classifier in `EmotionDetector` class

### Customizing Art Generation

- Modify `_generate_*_patterns()` methods in `ArtGAN` class
- Adjust color palettes in `emotion_mappings`
- Change pattern complexity and intensity parameters

## Dependencies

### Required
- `opencv-python`: Computer vision and image processing
- `numpy`: Numerical computing
- `pygame`: Graphics and audio

### Optional
- `dlib`: Advanced facial landmark detection
- `tensorflow`: Machine learning for emotion classification
- `pillow`: Image processing utilities

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Acknowledgments

- OpenCV for computer vision capabilities
- Pygame for graphics and audio
- The FER-2013 dataset for emotion classification research
- The artistic community for inspiration in abstract art generation

---

**Note**: This is a demonstration application. For production use, consider implementing more sophisticated emotion detection models and GAN architectures.
