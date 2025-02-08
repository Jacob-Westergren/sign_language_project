# Video Scene Extraction Tool

A Python-based tool for extracting and processing video scenes with interpreter detection.

## Overview

This tool processes video files to:
- Detect and extract scenes
- Identify sign language interpreters
- Generate cropped frames of interpreter segments
- Store scene metadata in JSON format

## Project Structure

```
project_root/
├── src/
│   ├── core/
│   │   ├── cloud_services/    # Cloud integration services
│   │   ├── scene.py          # Scene data model
│   │   ├── person.py         # Person detection model
│   │   ├── scene_extractor.py # Scene extraction logic
│   │   └── video_processor.py # Main video processing logic
│   └── __main__.py           # Entry point
├── data/
│   └── programs/             # Program video data
├── config.yaml               # Configuration settings
└── requirements.txt          # Project dependencies
```

## Prerequisites

### Python Environment
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### AWS Configuration
1. Install AWS CLI:
   ```bash  
   pip install awscli
   # or on macOS:
   brew install awscli
   ```

2. Configure AWS credentials:
   ```bash
   aws configure
   ```
   You will be prompted for:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region name (e.g., us-west-2)
   - Default output format (recommend: json)

### YOLO Model
1. Download the required YOLO model (yolov8s.pt) and place it in the project root.

## Configuration

1. Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```

2. Update `config.yaml` with your specific settings:
   - Data directories
   - Scene extraction parameters
   - Cloud service configurations

## Usage

1. Place your video files in the appropriate data directory:
   ```
   data/programs/<program_id>/<episode_id>/video.mp4
   ```

2. Run the processor:
   ```bash
   python -m src
   ```

3. Output will be generated in:
   - Scene metadata: `data/programs/<program_id>/<episode_id>/scenes.json`
   - Cropped frames: `data/programs/<program_id>/<episode_id>/scenes/`

## Output Format

### Scene JSON Structure
```json
{
    "scenes": [
        {
            "scene_id": 1,
            "start_frame": 0,
            "end_frame": 240,
            "interpreter_crop": [x1, y1, x2, y2],
            "interpreter_frequency": 5
        }
    ]
}
```

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
This project follows PEP 8 guidelines. Run linting with:
```bash
flake8 src/
```

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contact

[Your Contact Information]
