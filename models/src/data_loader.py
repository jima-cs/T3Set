import sys
from typing import Tuple, Optional, List, Union, Literal, Set, Dict, get_args
import re
import numpy as np
from typing import Tuple, Optional, List, Union, Literal
import torch
from sklearn.model_selection import train_test_split
import os
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from new_key_mapping import suggKey_to_dataKey_mapping, clean_key_mapping, AllowedSuggestionKey, \
    allowed_sugg_keys, AllowedSuggestionType
from key_dim import sensor_phy_dim_keys, RoundExpertSuggestions, StrokeExpertSuggestion, RoundDataIncludesPoseSensor, \
    PREDEFINE_pose_dim_keys, all_data_keys, ParsedSuggestion, PREDEFINE_sensor_dim_keys
from setting import USE_SENSOR_PROCESS

sample_case_path = '../sampledata'

def custom_collate_fn(batch):
    inputs, labels = zip(*batch)

    sensor_data, pose_data, stroke_mask_tensor = zip(*inputs)
    sensor_data = torch.stack(sensor_data) 
    pose_data = torch.stack(pose_data)  
    stroke_mask_tensor = torch.stack(stroke_mask_tensor)
    
    round_meta_info, sugg_key_type_set, expert_sugg_key_id_set = zip(*labels)
    round_meta_info = list(round_meta_info)
    sugg_key_type_set = list(sugg_key_type_set)
    expert_sugg_key_id_set = torch.stack(expert_sugg_key_id_set)

    return sensor_data, pose_data, stroke_mask_tensor, round_meta_info, sugg_key_type_set, expert_sugg_key_id_set
all_keys_in_raw_list_to_count = []
all_types_in_raw_list_to_count = []

def combine_expert_suggestions(expert_suggestions: List[RoundExpertSuggestions])-> RoundExpertSuggestions:
    combined_expert_suggestions: RoundExpertSuggestions = RoundExpertSuggestions(
        summary_suggestions=expert_suggestions[0].summary_suggestions + expert_suggestions[1].summary_suggestions,
        stroke_suggestions=expert_suggestions[0].stroke_suggestions + expert_suggestions[1].stroke_suggestions
    )
    return combined_expert_suggestions
def get_datakey_idx_from_datakey(key_: str) -> int:
    datakey_to_index = {key: idx for idx, key in enumerate(all_data_keys)}
    if key_ in datakey_to_index.keys():
        return datakey_to_index[key_]
    else:
        print('data key error', key_)
        return -1
def load_and_extract_sugg_from_object(sugg_object: List[Dict]) -> Tuple[List[ParsedSuggestion],List[AllowedSuggestionKey]]:
    sugg_list = []
    mentioned_keys: List[AllowedSuggestionKey] = []
    for comment in sugg_object:
        sugg_key = comment['suggestion_key']
        sugg_key = sugg_key.replace(' ', '_')
        if sugg_key not in allowed_sugg_keys:
            print('sugg key not in allowed sugg keys', sugg_key)

            sugg_key = 'others'
            continue
        if sugg_key == 'others':
            continue
        sugg_type = comment['suggestion_type']
        sugg_content = comment['comment_en']
        all_keys_in_raw_list_to_count.append(sugg_key)
        all_types_in_raw_list_to_count.append(sugg_type)
        suggestion = ParsedSuggestion(context=sugg_content, key=sugg_key, suggestion_type=sugg_type)
        sugg_list.append(suggestion)
        mentioned_keys.append(suggestion.key)
    return sugg_list, mentioned_keys
def load_and_extract_sugg_from_object_stroke(sugg_object: List[Dict]) -> Tuple[List[StrokeExpertSuggestion],List[AllowedSuggestionKey]]:
    sugg_list:List[StrokeExpertSuggestion] = []
    mentioned_keys: List[AllowedSuggestionKey] = []
    for comment in sugg_object:
        sugg_key = comment['suggestion_key']
        sugg_key = sugg_key.replace(' ', '_')
        if sugg_key not in allowed_sugg_keys:
            print('sugg key not in allowed sugg keys', sugg_key)
            sugg_key = 'others'
            continue
        sugg_type = comment['suggestion_type']
        sugg_content = comment['comment_en']
        all_keys_in_raw_list_to_count.append(sugg_key)
        all_types_in_raw_list_to_count.append(sugg_type)
        suggestion: ParsedSuggestion = ParsedSuggestion(context=sugg_content, key=sugg_key, suggestion_type=sugg_type)
        sugg_list.append(StrokeExpertSuggestion(stroke_index=int(comment['stroke_index']), suggestion=suggestion))
        mentioned_keys.append(suggestion.key)
    return sugg_list, mentioned_keys

def load_expert_suggestions(suggestion_path) -> List[RoundExpertSuggestions]:

    global_path_1 = os.path.join(suggestion_path, 'summary_suggestions/coach_comment_1.json')
    global_path_2 = os.path.join(suggestion_path, 'summary_suggestions/coach_comment_2.json')

    realtime_path_1 = os.path.join(suggestion_path, 'live_comments/coach_comment_1.json')
    realtime_path_2 = os.path.join(suggestion_path, 'live_comments/coach_comment_2.json')
    try:
        with open(global_path_1, 'r') as file:
            global_1_data_json = json.load(file)
        with open(global_path_2, 'r') as file:
            global_2_data_json = json.load(file)
        with open(realtime_path_1, 'r') as file:
            realtime_1_data_json = json.load(file)
        with open(realtime_path_2, 'r') as file:
            realtime_2_data_json = json.load(file)

    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {global_path_1} was not found.")
    except json.JSONDecodeError:
        raise ValueError(f"The file at {global_path_1} is not a valid JSON file.")
    global_comments_1, global_sugg_1_keys = load_and_extract_sugg_from_object(global_1_data_json)
    global_comments_2, global_sugg_2_keys = load_and_extract_sugg_from_object(global_2_data_json)
    realtime_comments_1, realtime_sugg_1_keys = load_and_extract_sugg_from_object_stroke(realtime_1_data_json)
    realtime_comments_2, realtime_sugg_2_keys = load_and_extract_sugg_from_object_stroke(realtime_2_data_json)
    
    combined_sugg_1 = list(set(global_sugg_1_keys + realtime_sugg_1_keys))
    

    e1 = RoundExpertSuggestions(summary_suggestions=global_comments_1,
                                stroke_suggestions=realtime_comments_1,
                                suggestion_keys=combined_sugg_1)
    combined_sugg_2 = list(set(global_sugg_2_keys + realtime_sugg_2_keys))
    e2 = RoundExpertSuggestions(summary_suggestions=global_comments_2,
                                stroke_suggestions=realtime_comments_2,
                                suggestion_keys=combined_sugg_2)
    return [e1, e2]


def calculate_frequencies(keys_list):
    
    frequency_dict = {}

    
    for key in keys_list:
        
        if key in frequency_dict:
            frequency_dict[key] += 1
        
        else:
            frequency_dict[key] = 1

   
    total_keys = len(keys_list)

    
    result_dict = {}

    for key, count in frequency_dict.items():
        
        frequency = count / total_keys
        result_dict[key] = {
            'count': count,
            'frequency': frequency
        }

   
    sorted_keys = sorted(result_dict.items(), key=lambda item: item[1]['count'], reverse=True)

    
    print(f"所有词频信息 in {total_keys}:")
    
    for i, (key, info) in enumerate(sorted_keys):
        print(f"{i + 1}. {key}: \t\t\t出现次数 = {info['count']}")


def load_round_data(data_dir=sample_case_path) -> List[RoundDataIncludesPoseSensor]:
    round_data_list:List[RoundDataIncludesPoseSensor] = []
    round_meta_info = {"player_id": "-1", "tech_name": "-1", "round_id": "-1"}
    for player_id in os.listdir(data_dir):
        
        round_meta_info['player_id'] = player_id
        player_dir = os.path.join(data_dir, player_id)
        if os.path.isdir(player_dir):
            for tech_name in os.listdir(player_dir):
                round_meta_info['tech_name'] = tech_name
                tech_dir = os.path.join(player_dir, tech_name)
                if os.path.isdir(tech_dir):
                    for round_id in ['round_00', 'round_01', 'round_02']:
                        round_meta_info['round_id'] = round_id
                        round_dir = os.path.join(tech_dir, round_id)
                        if os.path.isdir(round_dir):
                            
                            sensor_dir = os.path.join(round_dir, "sensor/matched_augmented_remake.json")
                            try:
                                with open(sensor_dir, 'r') as file:
                                    sensor_data_json = json.load(file)
                            except FileNotFoundError:
                                raise FileNotFoundError(f"The file at {sensor_dir} was not found.")
                            except json.JSONDecodeError:
                                raise ValueError(f"The file at {sensor_dir} is not a valid JSON file.")

                            extracted_sensor_data = []
                            extracted_sensor_data_aggregated = []
                            for stroke in sensor_data_json:
                                peak_moment = stroke.get("PeakMoment", {})
                                movements = stroke.get("movement",{})
                                windows_data_100 = movements.copy()
                                full_transposed_data = [[] for _ in sensor_phy_dim_keys]
                                for key_idx, key in enumerate(sensor_phy_dim_keys):
                                    for point in windows_data_100:
                                        full_transposed_data[key_idx].append(float(point[key]))
                                
                                full_transposed_data_np = np.array(full_transposed_data)
                                acc_peak_exp_sqrt_full = np.sqrt(full_transposed_data_np[0, :]**2 + full_transposed_data_np[1, :]**2 + full_transposed_data_np[2, :]**2)
                                agl_spd_peak_exp_sqrt_full = np.sqrt(full_transposed_data_np[3, :]**2 + full_transposed_data_np[4, :]**2 +full_transposed_data_np[5, :]**2)
                                agl_y_peak_full = full_transposed_data_np[7, :]
                                transposed_values_aggregated = [acc_peak_exp_sqrt_full, agl_y_peak_full, agl_spd_peak_exp_sqrt_full]
                                transposed_values = full_transposed_data
                                extracted_sensor_data.append(transposed_values_aggregated if USE_SENSOR_PROCESS else transposed_values)
                            
                            pose_dir = os.path.join(round_dir, "pose/matched.json")
                            try:
                                with open(pose_dir, 'r') as file:
                                    pose_data_json = json.load(file)
                            except FileNotFoundError:
                                raise FileNotFoundError(f"The file at {pose_dir} was not found.")
                            except json.JSONDecodeError:
                                raise ValueError(f"The file at {pose_dir} is not a valid JSON file.")
                            
                            extracted_pose_data = []
                            num_frames = 10  
                            num_joints = len(PREDEFINE_pose_dim_keys)
                            num_coordinates = 3  
                            for stroke in pose_data_json:
                                stroke_pose_data = np.zeros((num_joints, num_frames, num_coordinates))
                                stroke_frame_data = stroke.get("frames")
                                for frame_index, frame_data in enumerate(stroke_frame_data):
                                    if frame_index >= num_frames: 
                                        break
                                    landmarks = frame_data.get('landmarks', [])
                                    for landmark in landmarks:
                                        landmark_id, x, y, z = landmark
                                        if landmark_id in range(num_joints):
                                            stroke_pose_data[landmark_id, frame_index, :] = [x, y, z]
                                extracted_pose_data.append(stroke_pose_data)
                           
                            expert_sugg_keys_set = set()
                            suggestion_dir = os.path.join(round_dir, "comments")
                            suggestions: List[RoundExpertSuggestions] = load_expert_suggestions(suggestion_dir)

                            for sugg in suggestions:
                                for sk in sugg.suggestion_keys:
                                    expert_sugg_keys_set.add(allowed_sugg_keys.index(sk))
                            
                            min_len = len(extracted_pose_data)
                            if len(extracted_sensor_data) != len(extracted_pose_data):
                                print("len(extracted_sensor_data) != len(extracted_pose_data)", len(extracted_sensor_data), len(extracted_pose_data),round_meta_info)
                                min_len = min(len(extracted_sensor_data), len(extracted_pose_data))
                            max_stroke_num = 40
                            
                            if len(extracted_sensor_data) < max_stroke_num: 
                                padding_rows = max_stroke_num - len(extracted_sensor_data)
                                padding_data = np.zeros((padding_rows, len(extracted_sensor_data[0]), 100))
                                extracted_sensor_data.extend(padding_data.tolist())

                            if len(extracted_pose_data) < max_stroke_num:
                                padding_rows = max_stroke_num - len(extracted_pose_data)
                                fps = 10
                                padding_data = np.zeros((padding_rows, len(PREDEFINE_pose_dim_keys), fps, 3))
                                extracted_pose_data.extend(padding_data.tolist())
                           
                            stroke_mask = [1] * min_len + [0] * (max_stroke_num - min_len)
                            
                            round_data = RoundDataIncludesPoseSensor(
                                round_meta_info='-'.join(round_meta_info.values()),
                                pose_data=extracted_pose_data,
                                sensor_data=extracted_sensor_data,
                                stroke_mask=stroke_mask,
                                expert_suggestions=suggestions,
                                expert_sugg_key_id_set=expert_sugg_keys_set
                            )
                            round_data_list.append(round_data)
    print('finished loading')
    return round_data_list

def preprocess_data_as_data_loader(round_data_list_: List[RoundDataIncludesPoseSensor], batch_size: int, fps: int):
    print('len round_data_list:', len(round_data_list_))
    train_val_data, test_data = train_test_split(round_data_list_, test_size=0.2, random_state=42)
    train_data, valid_data = train_test_split(train_val_data, test_size=0.25, random_state=42) 

    def process_data(data_list: List[RoundDataIncludesPoseSensor]):
        sensor_list = []
        pose_list = []
        mask_list = []
        round_meta_info_list = []
        sugg_key_type_set_list = []
        label_matrix = np.zeros((len(data_list), len(allowed_sugg_keys)), dtype=np.float32)
        for idx, data in enumerate(data_list): 
            
            sensor_np = np.array(data.sensor_data, dtype=np.float32)
            pose_np = np.array(data.pose_data, dtype=np.float32)
            mask_np = np.array(data.stroke_mask, dtype=np.int32)
            all_sugg_types_from_exp: Set[Tuple[AllowedSuggestionKey, AllowedSuggestionType]] = set()
            
            for round_exp_sugs in data.expert_suggestions:
                for stroke_exp_sug in round_exp_sugs.stroke_suggestions:
                    sugg_type = stroke_exp_sug.suggestion.suggestion_type
                    sugg_key = stroke_exp_sug.suggestion.key
                    all_sugg_types_from_exp.add((sugg_key, sugg_type))
                for summary_exp_sug in round_exp_sugs.summary_suggestions:
                    sugg_type = summary_exp_sug.suggestion_type
                    sugg_key = summary_exp_sug.key
                    all_sugg_types_from_exp.add((sugg_key, sugg_type))


            
            sensor_np = sensor_np.reshape(-1, len(PREDEFINE_sensor_dim_keys), 100)
            pose_np = pose_np.reshape(-1, len(PREDEFINE_pose_dim_keys), fps, 3)

            sensor_list.append(sensor_np)
            pose_list.append(pose_np)
            mask_list.append(mask_np)
            round_meta_info_list.append(data.round_meta_info)
            sugg_key_type_set_list.append(all_sugg_types_from_exp)
            
            if 24 in data.expert_sugg_key_id_set and len(data.expert_sugg_key_id_set) > 1: 
                data.expert_sugg_key_id_set.remove(24)
            label_matrix[idx, [key_id for key_id in data.expert_sugg_key_id_set]] = 1.0
        sensor_tensors = [torch.from_numpy(x) for x in sensor_list]
        pose_tensors = [torch.from_numpy(x) for x in pose_list]
        mask_tensors = [torch.from_numpy(x) for x in mask_list]
        label_tensor = torch.from_numpy(label_matrix)

        return list(zip(sensor_tensors, pose_tensors, mask_tensors)), list(zip(round_meta_info_list, sugg_key_type_set_list, label_tensor))

    train_inputs, train_labels = process_data(train_data)
    valid_inputs, valid_labels = process_data(valid_data)
    test_inputs, test_labels = process_data(test_data)

    class RoundPoseSensorDataset(Dataset):
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return self.inputs[idx], self.labels[idx]

    train_dataset = RoundPoseSensorDataset(train_inputs, train_labels)
    valid_dataset = RoundPoseSensorDataset(valid_inputs, valid_labels)
    test_dataset = RoundPoseSensorDataset(test_inputs, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=custom_collate_fn, pin_memory=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True, num_workers=8)
    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    load_round_data(sample_case_path)
    calculate_frequencies(all_keys_in_raw_list_to_count)
    calculate_frequencies(all_types_in_raw_list_to_count)