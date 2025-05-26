import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import json
import os 

def downsample_data(data, original_fps, target_fps):
    step = int(original_fps / target_fps)
    return data[::step]

def get_peak_window_data(data, peak_index, fps, window_duration=0.5):
    samples_per_second = fps
    samples_per_window = int(window_duration * samples_per_second)
    start_idx = max(0, peak_index - samples_per_window)
    end_idx = min(len(data), peak_index + samples_per_window + 1)
    return downsample_data(data[start_idx:end_idx],60,10)



def smooth_data(data, window_size=5):
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return smoothed

def my_find_peaks(data,distance = 30):
    max_height_of_stroke_peak = max(data)
    min_height_of_stroke_peak = min(data)

    peaks, _ = find_peaks(data, height=min_height_of_stroke_peak+(0.6)*(max_height_of_stroke_peak-min_height_of_stroke_peak), distance=distance)

    return peaks
def get_pose_10hz_6lm(pose_data,output_path):
    right_wrist_x = []
    right_wrist_y = []
    for frameIndex,frame in enumerate(pose_data):
        for landmark in frame['landmarks']:
            if landmark[0] == 16: 
                right_wrist_x.append(landmark[1])
                right_wrist_y.append(landmark[2])

    smoothed_right = smooth_data(np.array(right_wrist_x))
    if output_path.split('/')[-4] != 'pendulum':
        peaks_right = my_find_peaks(smoothed_right)
        print("len of peaks",len(peaks_right))
    else:
        peaks_right = my_find_peaks(smoothed_right,distance = 150)
        print("len of peaks",len(peaks_right))
    data_10_selected_joints = [{"stroke_index":peak_idx,"frames":get_peak_window_data(pose_data, peak, fps=60)} for peak_idx,peak in  enumerate(peaks_right)]

    with open(output_path, 'w') as of:
        json.dump(data_10_selected_joints, of)
    print(output_path)
    plt.figure(figsize=(14, 7))

    plt.subplot(2, 1, 2)
    plt.plot(right_wrist_x, label='Original Right Wrist')
    plt.plot(smoothed_right, label='Smoothed Right Wrist', color='red')
    plt.scatter(peaks_right, smoothed_right[peaks_right], marker='x', color='green', label='Peaks')
    plt.title('Right Wrist Position Over Time with Peaks')
    plt.xlabel('Frame')
    plt.ylabel('X Position')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(output_path.replace('.json', '.png'))

def pose_stroke_extraction(root_path):
    for root,dirs,files in os.walk(root_path):
        for file_name in files:
            if file_name == "pose.json" and os.path.basename(root) == 'pose':
                with open(os.path.join(root,file_name),'r') as f:
                    pose_data = json.load(f)
                    output_path = os.path.join(root,'downsampled_peak_windows_right.json')
                    get_pose_10hz_6lm(pose_data,output_path)

if __name__ == '__main__':
    root_path = "../../sampledata"  #replace with your own dataset path
    pose_stroke_extraction(root_path)