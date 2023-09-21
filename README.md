# WALDO-Enhanced-Aerial-Detection

## Overview

This repository builds upon the [WALDO](https://github.com/stephansturges/WALDO) object detection AI model to offer enhanced capabilities for analyzing aerial footage. In addition to the object annotations provided by WALDO, this project also generates an enhanced 2D map to give a bird's-eye perspective of the detected objects. This enables a transformative viewpoint that can serve multiple use-cases including urban planning, real-time traffic monitoring, and aerial surveillance.

## Features

- Real-time object detection and annotation in aerial footage.
- Enhanced 2D mapping with icons color-coded to match WALDO's detection labels.
- Output video saving functionality.
  
## Requirements

- Python 3.x
- OpenCV
- ONNX Runtime
- NumPy
- Other dependencies are listed in `requirements.txt`

## Setup

1. Clone this repository.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your aerial footage videos in the `./input_vids` folder.
4. Add the main.py script in the `playground` folder.
5. Run the main script:
   ``` bash
   python3 your_script_name.py
   ```

## Acknowledgments

This project leverages the powerful WALDO object detection model. A huge shoutout to the original creators of WALDO for their exceptional work.

## License

This project is open-source and available under the [MIT License](LICENSE).
