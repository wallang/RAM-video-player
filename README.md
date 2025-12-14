# RAM Video Player

A RAM-based video player with Quality Control (QC) annotation features. Built with PySide6 and OpenCV, this application loads entire videos into memory for smooth playback and frame-accurate drawing annotations.

## Features

- **RAM-based playback**: Load entire videos into memory for instant frame access
- **Frame-accurate navigation**: Step through frames with keyboard shortcuts
- **Drawing tools**: Annotate frames with multiple colors and adjustable stroke widths
- **IN/OUT points**: Mark specific frame ranges for review
- **Timeline visualization**: Visual timeline with frame markers and annotation indicators
- **Pixel inspector**: Real-time pixel value display (normalized RGB values)
- **Export capabilities**: Export current frame or all annotated frames as PNG
- **Keyboard shortcuts**: Efficient workflow with Space (play/pause), arrows (step frames), I/O (set IN/OUT points)
- **Changelog tracking**: Built-in version history

## Requirements

- Python 3.8+
- Virtual environment support (venv)
- Sufficient RAM for video files (entire video is loaded into memory)

## Installation

### 1. Clone or navigate to the project directory

```bash
cd /path/to/ram-video-player
```

### 2. Create a Python virtual environment

On **macOS/Linux**:
```bash
python3 -m venv .venv
```

On **Windows**:
```bash
python -m venv .venv
```

### 3. Activate the virtual environment

On **macOS/Linux**:
```bash
source .venv/bin/activate
```

On **Windows (Command Prompt)**:
```bash
.venv\Scripts\activate.bat
```

On **Windows (PowerShell)**:
```bash
.venv\Scripts\Activate.ps1
```

After activation, your terminal prompt should show `(.venv)` prefix.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `numpy` - Numerical operations and array handling
- `opencv-python` - Video reading and image processing
- `PySide6` - Qt bindings for the GUI
- `psutil` (optional) - RAM usage display

## Running the Application

With the virtual environment activated:

```bash
python video.py
```

Or make it executable (macOS/Linux):
```bash
chmod +x video.py
./video.py
```

## Usage

1. **Open Video**: Click "Videó megnyitása" to select a video file (supports .mov, .mp4, .m4v, .avi)
2. **Playback Controls**:
   - Space: Play/Pause
   - Left/Right Arrow: Step frame backward/forward
   - Play buttons: Forward/reverse playback
3. **Drawing**:
   - Click a color swatch to enable drawing mode
   - Adjust stroke width with the slider
   - Draw on frames with mouse
   - Ctrl+Z: Undo last stroke
4. **IN/OUT Points**:
   - I: Set IN point
   - O: Set OUT point
   - Playback will loop within the IN/OUT range
5. **Export**:
   - "PNG (aktuális)": Save current frame with annotations
   - "Export annotált PNG-k...": Export all annotated frames

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| Left Arrow | Previous frame |
| Right Arrow | Next frame |
| I | Set IN point |
| O | Set OUT point |
| V | Toggle QC overlay visibility |
| Page Up | Jump to previous annotated frame |
| Page Down | Jump to next annotated frame |
| Ctrl+Z | Undo last stroke |

## Deactivating Virtual Environment

When done, deactivate the virtual environment:

```bash
deactivate
```

## Troubleshooting

### Module Not Found Errors
If you get `ModuleNotFoundError`, ensure:
1. Virtual environment is activated (check for `(.venv)` in prompt)
2. Dependencies are installed: `pip install -r requirements.txt`

### Memory Issues
This player loads entire videos into RAM. If you experience memory errors:
- Close other applications
- Try shorter or lower resolution videos
- The app will display available RAM in the status bar

### Video Won't Load
Ensure the video file is:
- In a supported format (.mov, .mp4, .m4v, .avi)
- Not corrupted
- Readable by OpenCV

## Development

Current version: **v0.016**

To view the full changelog, click "Változások..." button in the application.

## License

This project uses:
- PySide6 (LGPL)
- OpenCV (Apache 2.0)
- NumPy (BSD)


