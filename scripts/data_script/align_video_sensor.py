import os
import json
import cv2
import re
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from scipy.spatial.distance import euclidean
def match_values(x, y, delta):
    matched_x = []
    matched_y = []
    matched_x_index = []
    matched_y_index = []
    for y_j in y:
        candidates = [x_i for x_i in x if y_j - delta <= x_i <= y_j + delta]
        if len(candidates) == 1 and matched_x.count(candidates[0]) == 0:
            matched_x.append(candidates[0])
            matched_y.append(y_j)
            matched_x_index.append(list(x).index(candidates[0]))
            matched_y_index.append(list(y).index(y_j))

    return matched_x, matched_y,matched_x_index,matched_y_index

def traversal_match(x, y, delta=0.3,max_matched =0):
    offset_options = np.linspace(-1, 1, 20000)
    best_offset = None
    for offset in offset_options:
        matched_x, matched_y,_,_ = match_values(x+offset, y, delta)
        if len(matched_x) > max_matched:
            max_matched = len(matched_x)
            best_offset = offset
    return best_offset

def calculate_loss(p,s,offset):
    p_pred = s + offset
    return np.sum((p - p_pred) ** 2)


def calculate_offset(x, y, w=1,partial_len = 5, length = 10,learning_rate=0.01, iterations=1000):
    x = np.array(x)
    y = np.array(y)
    
    mid_index = x.shape[0] // 2
    start = mid_index - length // 2
    end = mid_index + length // 2
    x_subset = x[start:end]
    best_b = None
    min_mse = float('inf')
    max_matched = 0
    for i in range(y.shape[0] - length+1):
        y_subset = y[i: i + partial_len] 
        
        for j in range(partial_len+1): 
            x_sub = x_subset[j: j + partial_len]
            b = 0
            for _ in range(iterations):
                y_pred = w * x_sub + b
                gradient = -2 * np.sum(y_subset - y_pred) / y_subset.shape[0]
                b -= learning_rate * gradient
            
            mse = np.mean((y_subset - (w * x_sub + b)) ** 2)
            if best_b ==None:
                best_b = b
                min_mse = mse
            
            matched,_,_,_ = match_values(x+b, y, delta=0.3)
            if len(matched) > max_matched and abs(b) < 1.5:
                max_matched = len(matched)
                best_b = b
                min_mse = mse
    print(f"best_b: {best_b}, min_mse: {min_mse}")
    return best_b, min_mse

def extract_sensor_time(folder_path,video_file):
    result = []
    video_time = re.match(r"(\d{4}-\d{2}-\d{2}) (\d{2}-\d{2}-\d{2})", video_file)
                
    if video_time:
        video_time_str = video_time.group(2)
        video_datetime_str = video_time.group(1)+" " +video_time.group(2)
        video_time = datetime.strptime(video_time_str, "%H-%M-%S")
        video_datetime = datetime.strptime(video_datetime_str,"%Y-%m-%d %H-%M-%S")
        json_path = os.path.join(folder_path, 'sensor', 'processed.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as json_file:
                processed_data = json.load(json_file)
                for entry in processed_data:
                    imu_time = entry.get("stroke_time_on_imu")
                    imu_datetime = datetime.strptime(imu_time, "%H:%M:%S.%f")
                    time_diff = imu_datetime - video_time
                    
                    result.append(time_diff.total_seconds())
    return result,video_datetime, imu_datetime

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

def extract_video_pose_time(file1_path, file2_path, video_path):
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
            print(f"未能在文件2中找到 stroke_index {stroke_index} 的帧位置。")
            continue
        
        middle_frame_index = (first_frame_index + last_frame_index) // 2
        
        time_in_video = calculate_time(middle_frame_index, frame_rate)
        stroke_time_list.append(time_in_video)
    return stroke_time_list

def dump_matched_data(origin_path_sensor, matched_x_index, output_path_sensor, origin_path_pose, matched_y_index, output_path_pose, matched_x,matched_y,start_time_video,imu_date_time):
    data = None
    matched_data_x = []
    matched_data_y = []
    with open(origin_path_sensor, 'r') as file:
        data = json.load(file)
        matched_data_x = [data[i] for i in matched_x_index]

    with open(origin_path_pose, 'r') as file:
        data = json.load(file)
        matched_data_y = [data[i] for i in matched_y_index]

    start_time_video_timestamp = start_time_video.timestamp()
    start_time_video = datetime.fromtimestamp(start_time_video_timestamp)
    
    for i in range(len(matched_data_y)):
        stroke_time_in_video =datetime.fromtimestamp(matched_y[i] + start_time_video_timestamp).strftime("%Y:%m:%d %H:%M:%S.%f")[:-3]
        stroke_time_on_imu = datetime.fromtimestamp(matched_x[i] + start_time_video_timestamp).strftime("%Y:%m:%d %H:%M:%S.%f")[:-15]+matched_data_x[i].get("stroke_time_on_imu")
        
        matched_data_x[i]["stroke_time_on_imu"] = stroke_time_on_imu
        matched_data_y[i]["stroke_time_in_video"] = stroke_time_in_video

        matched_data_x[i]["start_time_video"] = start_time_video.strftime("%Y:%m:%d %H:%M:%S.%f")[:-3]
        matched_data_y[i]["start_time_video"] = start_time_video.strftime("%Y:%m:%d %H:%M:%S.%f")[:-3]
        
        matched_data_x[i].pop("strike_index")
        matched_data_x[i]["stroke_index"] = i
        matched_data_y[i]["stroke_index"] = i
        
    with open(output_path_sensor, 'w') as of:
        json.dump(matched_data_x, of)
    with open(output_path_pose, 'w') as of:
        json.dump(matched_data_y, of)

def plot_matched_data(matched_x,matched_y,image_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(matched_x)), matched_x, label='Matched Sensor Time')
    plt.plot(range(len(matched_y)), matched_y, label='Matched Pose Time')
    plt.xlabel('Index')
    plt.ylabel('Time (s)')
    plt.title('Matched Sensor and Pose Times')
    plt.legend()
    plt.savefig(image_path)
    plt.close()

matched_list = []
def match_pose_sensor(root_dir):
    for folder_path, subfolders, files in os.walk(root_dir):
        if os.path.basename(folder_path).startswith('round'):
            video_dir = os.path.join(folder_path, 'video', 'remote')
            if os.path.exists(video_dir):
                video_files = [f for f in os.listdir(video_dir) if f.endswith('.mkv') or f.endswith('.mp4')]
                video_file = video_files[0]
                sensor_time,start_time_video,imu_date_time = extract_sensor_time(folder_path, video_file)

                round_path = folder_path
                file1_path = os.path.join(round_path, "pose/downsampled_peak_windows_right.json")
                file2_path = os.path.join(round_path, "pose/pose.json")
                pose_time = extract_video_pose_time(file1_path, file2_path, os.path.join(video_dir, video_file))
                if(len(sensor_time)>10 and len(pose_time)>15):
                    offset,_ = calculate_offset(sensor_time, pose_time,length=10)
                    matched_x, matched_y, matched_x_index, matched_y_index = match_values(sensor_time+offset,pose_time,delta=0.3)
                    output_path_x = os.path.join(round_path, 'sensor', 'matched.json')
                    output_path_y = os.path.join(round_path, 'pose', 'matched.json')
                    output_image_path = os.path.join(round_path,'sensor', 'matched.png')
                    dump_matched_data(os.path.join(round_path, 'sensor', 'processed.json'), matched_x_index, output_path_x, \
                                        os.path.join(round_path, 'pose', 'downsampled_peak_windows_right.json'), matched_y_index, output_path_y, matched_x,matched_y,\
                                            start_time_video,imu_date_time)
                    plot_matched_data(matched_x, matched_y, output_image_path)
                    matched_list.append(len(matched_x))
                    print(folder_path)
                    print(sensor_time)
                    print(pose_time)
                    print(len(matched_x))

    print(f"matched: {matched_list}")


if __name__ == "__main__":
    target_directory = "../sampledata" 
    match_pose_sensor(target_directory)
