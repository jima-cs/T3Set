# Building the T3Set
## The data processing pipeline
Please refer to the Chapter 3 of our paper for more details.

## Scripts
- `extract_pose_using_mediapipe.py`: extract pose using mediapipe
- `stroke_detection.py`: detect strokes from the round data
- `align_video_sensor.py`: align video and sensor data
- `add_stroke_sequence_timestamp.py`: add stroke sequence timestamp
- `structuralize_suggestion.py`: structuralize suggestion from the text which is transformed from coach's audio

The script order can not completely demonstrate the data construction and processing flow. 
For a more clear interpretation, please refer to Chapter 3 of the paper.
# License
CC BY-NC-ND 4.0