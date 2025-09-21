# ALT-DJ: AI-Powered DJ Performance Analysis and Style Transfer

## Overview

ALT-DJ is a comprehensive machine learning project for analyzing DJ performances, extracting audio-visual correlations, and performing style transfer between different DJ mixing styles. The system combines computer vision, audio processing, and neural networks to understand and recreate DJ mixing techniques.

## Features

- **DJ Performance Analysis**: Automated detection and analysis of DJ equipment interactions using YOLO object detection
- **Audio-Visual Correlation**: Maps visual DJ actions to corresponding audio changes
- **Style Transfer**: Neural network-based audio style transfer between different DJ styles
- **Interactive Player**: Desktop application for real-time DJ performance simulation
- **Video Clipping**: Automated extraction of performance highlights based on audio-visual events

## Project Structure

```
train-2/
├── Core Modules
│   ├── altdj_audio_analyzer.py      # Audio feature extraction and analysis
│   ├── altdj_neural_network.py      # Core neural network implementation
│   ├── altdj_desktop_app.py         # Main GUI application
│   └── dj_analyzer.py               # DJ performance analysis tools
│
├── Style Transfer
│   ├── audio_style_transfer_nn.py   # Style transfer neural network
│   ├── audio_style_gan.py          # GAN-based style transfer
│   └── modernaudiodataset.py       # Dataset handling for training
│
├── Media Processing
│   ├── altdj_video_clipper.py      # Video clip extraction
│   ├── altdj_interactive_player.py  # Interactive playback
│   └── altdj_fixed_player.py       # Fixed playback mode
│
├── Models (included)
│   ├── altdj_trained_model.pt      # Core trained model (60KB)
│   ├── audio_style_transfer_model.pt # Style transfer model (2.5MB)
│   ├── yolo11n.pt                  # YOLO v11 nano (5.6MB)
│   └── yolov8n.pt                  # YOLO v8 nano (6.5MB)
│
└── Utilities
    ├── kaggle_audio_downloader.py  # Audio dataset downloader
    └── train_and_analyze.py        # Training pipeline
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- FFmpeg (for audio/video processing)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/alt-dj.git
cd alt-dj
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download large model files (optional, for full GAN functionality):
   - Large GAN models are not included due to size (>250MB each)
   - Contact repository owner for access to pre-trained GAN checkpoints

## Usage

### Running the Desktop Application

```bash
python altdj_desktop_app.py
```

The GUI application provides:
- Audio file loading and playback
- Real-time visualization
- Style transfer application
- Performance analysis

### Training New Models

1. Prepare your training data (DJ performance videos)
2. Run the analysis pipeline:
```bash
python train_and_analyze.py
```

3. Train the style transfer model:
```bash
python audio_style_transfer_nn.py
```

### Analyzing DJ Performances

```bash
python dj_analyzer.py --video path/to/dj_performance.mp4
```

This will:
- Detect DJ equipment and hand positions
- Extract audio features
- Generate correlation mappings
- Output analysis results to JSON

## Key Components

### Audio Analysis
The `ALTDJAudioAnalyzer` class extracts:
- Spectral features (centroid, rolloff, contrast)
- Rhythm features (tempo, beat tracking)
- Energy and dynamics
- Frequency band analysis

### Visual Analysis
Using YOLO models to detect:
- DJ equipment (turntables, mixers, controllers)
- Hand positions and gestures
- Equipment interactions

### Neural Networks
- **Style Transfer NN**: Convolutional architecture for audio style transformation
- **GAN Implementation**: Generative adversarial network for advanced style transfer
- **Correlation Network**: Maps visual features to audio changes

## Data Format

### Input
- **Video**: MP4 format, 30fps recommended
- **Audio**: WAV format, 44.1kHz sample rate
- **Training Data**: JSON files with timestamp-aligned audio-visual features

### Output
- **Analysis Results**: JSON with detected events and correlations
- **Styled Audio**: WAV files with applied style transfer
- **Video Clips**: Extracted performance highlights

## Model Performance

- Object detection: ~85% accuracy on DJ equipment
- Style transfer: Preserves rhythm while transforming tonal characteristics
- Real-time processing: 30fps for video analysis, <100ms latency for audio

## Dependencies

Core libraries required:
- PyTorch (deep learning)
- Librosa (audio analysis)
- OpenCV (computer vision)
- NumPy/Pandas (data processing)
- Pygame (audio playback)
- Tkinter (GUI)

See `requirements.txt` for complete list with versions.

## Limitations

- Large GAN models not included in repository (>1GB total)
- Requires significant GPU memory for training (>8GB recommended)
- Analysis accuracy depends on video quality and lighting

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## License

This project is for educational purposes as part of AME 534 - ML Media Arts course.

## Acknowledgments

- YOLO models from Ultralytics
- Audio processing techniques from Librosa documentation
- DJ performance datasets from various sources

## Contact

For questions about large model files or dataset access, please open an issue on GitHub.

---

**Note**: This is a machine learning research project. The style transfer and analysis results are approximations and may not perfectly replicate professional DJ techniques.