import json
import cv2
import os

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def find_frame_index(landmarks_list, target_landmarks):
    for index, frame in enumerate(landmarks_list):
        if frame['landmarks'] == target_landmarks:
            return index
    return None

def calculate_time(frame_number, frame_rate):
    return frame_number / frame_rate

from datetime import datetime
def process_matched_files(pose_file_path, comment_path):
    print(f"Processing {pose_file_path}")
    pose_file_data = load_json(pose_file_path)
    stroke_time_list = []
    for stroke in pose_file_data:
        stroke_time = datetime.strptime(stroke['stroke_time_in_video'], '%Y:%m:%d %H:%M:%S.%f')
        video_start = datetime.strptime(stroke['start_time_video'], '%Y:%m:%d %H:%M:%S.%f')
        time_in_video = stroke_time - video_start
        stroke_index = stroke['stroke_index']
        stroke_time_list.append({"stroke_index": stroke_index, "time_in_video": time_in_video.total_seconds()})
    
    comments_folder = comment_path
    
    for root, _, files in os.walk(comments_folder):
        for comment_file in files:
            if comment_file.endswith('.json'):
                print(f"Processing {comment_file}")
                comment_file_path = os.path.join(root, comment_file)
                comments_data = load_json(comment_file_path)
            
                match_comment(root,comments_data, stroke_time_list)
                with open(comment_file_path, 'w') as file:
                    json.dump(comments_data, file, ensure_ascii=False, indent=4)


def process_files(file1_path, file2_path, video_path,comment_path):
    print(f"Processing {file1_path} and {file2_path}")
    file1_data = load_json(file1_path)
    file2_data = load_json(file2_path)
    
    video = cv2.VideoCapture(video_path)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    stroke_time_list = []
    for stroke in file1_data:
        stroke_index = stroke['stroke_index']
        frames = stroke['frames']
        
        first_frame_landmarks = frames[0]['landmarks']
        last_frame_landmarks = frames[-1]['landmarks']
        
        first_frame_index = find_frame_index(file2_data, first_frame_landmarks)
        last_frame_index = find_frame_index(file2_data, last_frame_landmarks)
        
        if first_frame_index is None or last_frame_index is None:
            continue
        
        middle_frame_index = (first_frame_index + last_frame_index) // 2
        
        time_in_video = calculate_time(middle_frame_index, frame_rate)
        stroke_time_list.append({"stroke_index": stroke_index, "time_in_video": time_in_video}) 
    video.release()
    comments_folder = comment_path
    
    for root, _, files in os.walk(comments_folder):
        for comment_file in files:
            if comment_file.endswith('.json'):
                print(f"Processing {comment_file}")
                comment_file_path = os.path.join(root, comment_file)
                comments_data = load_json(comment_file_path)
                
                if not root.endswith('processed'):
                    match_comment(root,comments_data, stroke_time_list)
                else:
                    comments_data = comments_data["comments"]
                    match_comment(root,comments_data, stroke_time_list)
                with open(comment_file_path, 'w') as file:
                    json.dump(comments_data, file, ensure_ascii=False, indent=4)

def match_comment(parent_folder,comments_data, stroke_time_list):
    for comment in comments_data:
        time_parts = comment['start'].split(':')
        minutes = int(time_parts[1])
        seconds_parts = time_parts[2].split(',')
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1])
        start_time = minutes * 60 + seconds + milliseconds / 1000.0
        
        valid_strokes = [x for x in stroke_time_list if x['time_in_video'] <= start_time]
        if valid_strokes:
            closest_stroke = max(valid_strokes, key=lambda x: x['time_in_video'])
        else:
            closest_stroke = stroke_time_list[0]
        comment['stroke_index'] = closest_stroke['stroke_index'] if closest_stroke else None
        if 'global' in parent_folder:
            comment['type'] = 'global'
        else:
            comment['type'] = 'realtime'

def find_stroke(root_path):
    base_path = root_path
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name.startswith("round"):
                round_path = os.path.join(root, dir_name)
                file1_path = os.path.join(round_path, "pose/downsampled_peak_windows_right.json")
                file2_path = os.path.join(round_path, "pose/pose.json")
                comment_path = os.path.join(round_path, "comments")
                video_files = [f for f in os.listdir(os.path.join(round_path, "video/remote")) if f.endswith(".mkv") or f.endswith(".mp4")]
                if video_files:
                    video_path = os.path.join(round_path, "video/remote", video_files[0])
                else:
                    video_path = None
                if os.path.exists(file1_path) and os.path.exists(file2_path) and os.path.exists(video_path):
                    pose_file_path = os.path.join(round_path, "pose/matched.json")
                    process_matched_files(pose_file_path,comment_path)

if __name__ == "__main__":
    root_path = "../../sampledata"
    find_stroke(root_path)