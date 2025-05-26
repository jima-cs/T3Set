# -*- coding: utf-8 -*-

"""
File Name: eval_vllm.py
Author: Ji Ma
Date Created: 2025-02-11
Last Modified: 2025-02-26
Description: This script performs evaluation for video LLM of our baseline. Its target is to evaluate the different vLLM ability to select the suggestion key, select the suggestion (key,type), generating content.
WARNING: Please make sure to set the `FLAG_SENSECOACH=False` in `config.py`
"""
import os
import json
import re
import sys
import numpy as np
import time
from datetime import datetime
from openai import OpenAI
from google import genai
from typing import List, Union, Literal, Tuple, Set
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
import threading
import requests
import tqdm
from json import JSONDecodeError
from config import top_k
from eval_textonly import SuggestionUnit, evaluate_suggestion_f1, calc_key, calc_keytype, calc_keytype_after_correct_s1l
from new_key_mapping import AllowedSuggestionKey, AllowedSuggestionType

data_load_lock = threading.Lock()
# current_script_path = os.path.abspath(__file__)
# scripts_dir = os.path.dirname(current_script_path)
# project_root = os.path.dirname(scripts_dir)
# sys.path.insert(0, project_root)

from data_loader_eval import load_round_data, sample_case_path
from key_dim import PREDEFINE_sensor_dim_keys, PREDEFINE_pose_dim_keys, RoundDataIncludesPoseSensor


class SuggestionsLLM(BaseModel):
    suggestions_list: List[SuggestionUnit]


def format_floats_in_list(item):
    if isinstance(item, float):
        return f"{item:.3f}"
    elif isinstance(item, list):
        return [format_floats_in_list(subitem) for subitem in item]
    else:
        return item


def load_llm_client(llm_name):
    if llm_name in ['gpt-4o', 'o1-mini']:
        api_key = os.getenv("OPENAI_API_KEY")
        openai_client = OpenAI(
            api_key=api_key
        )
        return openai_client
    elif llm_name == 'gemini':
        api_key = "blank"
        gemini_client = genai.Client(
            api_key=api_key,
        )
        return gemini_client



def fix_json_string(wrong_json_string):
    if wrong_json_string.startswith("['") and wrong_json_string.endswith("']"):
        wrong_json_string = wrong_json_string[2:-2]  # 移除最外层的单引号和括号

    if not wrong_json_string.endswith('}'):
        wrong_json_string += '}'

    return wrong_json_string


def process_json_strings(input_strings):
    suggestions_by_llm = {
        'suggestions_list': []
    }

    for wrong_json_string in input_strings:
        corrected_string = fix_json_string(wrong_json_string)

        try:
            # 尝试将修正后的字符串加载为 JSON
            json_data = json.loads(corrected_string)

            # 将 suggestions_list 添加到 suggestions_by_llm 中
            suggestions_by_llm['suggestions_list'] = json_data['suggestions_list']

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")

    return suggestions_by_llm


def parse_llm_response(llm_response) -> dict:
    try:
        # 提取响应文本
        response_text = llm_response.candidates[0].content.parts[0].text

        # 提取JSON内容（兼容多行和```标记）
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if not json_match:
            raise ValueError("未找到有效的JSON数据块")

        json_str = json_match.group(1)
        suggestions_data = json.loads(json_str)

        # 确保数据格式符合预期
        if "suggestions_list" not in suggestions_data:
            raise ValueError("响应中缺少 `suggestions_list` 字段")

        return suggestions_data

    except (AttributeError, KeyError, IndexError) as e:
        print(f"响应结构解析失败: {str(e)}")
        return {"suggestions_list": []}  # 返回空结构避免后续崩溃
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {str(e)}")
        return {"suggestions_list": []}
    except Exception as e:
        print(f"未知错误: {str(e)}")
        return {"suggestions_list": []}


def extract_json_from_candidates(candidates):
    json_text = candidates[0].content.parts[0].text
    print('json_text', json_text)
    json_text = json_text.strip().replace('```json\n', '').replace('```', '')
    print('json_text', json_text)
    json_data = json.loads(json_text)
    suggestions_list = json_data.get('suggestions_list', [])
    output_dict = {
        "suggestions_list": [
            {
                "suggestion_key": suggestion.get('suggestion_key'),
                "suggestion_type": suggestion.get('suggestion_type')
            }
            for suggestion in suggestions_list
        ]
    }

    return output_dict


def generate_sugg_one_round(dataprompt, llm_chat_client, topk, video_path, llm_name='gemini'):
    prompt = f"""
    You are a table tennis coach. I have recorded video data and sensor data of a table tennis technique training session. 
    This session is called 'multi-ball' training, meaning the player strikes the ball using the same technique repeatedly for N times in a round.
    Your task is to analyze the data I provide, each with a video and a sequence of the corresponding sensor data, 
    describing the behavior of the player around the stroke moments. You will also know the technique of the stroke as well.
    <data-description>
    Here is the explanation of the sensor data:
        - acc_x: Acceleration along the x-axis of the imu
        - agl_speed_x: Angular speed along the x-axis of the imu
        - agl_x: Angle along the x-axis of the imu
        - mgt_x: Magnetic field along the x-axis of the imu
        - quat_1: Quaternion component 1
    </data-description>
    <task>
    Based on the data, you need to provide the most important {topk} suggestions for the player to improve his technique. 
    Each suggestion must describe only one aspect of the player's behavior and how to improve it.
    The suggestions must contain 3 parts: 'content', 'suggestion_key' and 'suggestion_type'.
    The 'suggestion_key' part in each suggestion must be unique and should be one of the predefined types.
    </task>
    <task-requirements>
    The 'content' part should be a sentence describing the suggestion.

    The 'suggestion_key' part should be the key of the data that the suggestion is based on. The predefined types for'suggestion_key' are: 
    ['center_of_gravity', 'right_wrist', 'right_shoulder', 'right_forearm', 'time_of_striking_ball',    
    'angle_of_racket', 'left_foot', 'right_foot', 'waist', 'right_upper_arm', 'left_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'backswing_of_racket', 'left_hand', 'right_hand',
    'left_leg', 'right_leg', 'left_knee', 'right_knee', 'right_finger', 'upper_body', 'grip_of_racket',
    'others'], you must not create a new one.

    The 'suggestion_type' part should be the type of the suggestion. The predefined types for 'suggestion_type' are:
    Based on this information, you need to:
    - position: Indicates that the position of this key is incorrect.
    - strength: Indicates that the force exertion or explosiveness of this key is incorrect or insufficient.
    - stability: Indicates that the stability of this key is incorrect.
    - racket: Indicates that the relationship between this key and the racket is incorrect.
    - others: Represents other types of suggestions related to this key.

    Please ensure that the output is in the JSON format as shown in the example:

    <output format>{{"suggestions_list":[{{"content":"suggestion content","suggestion_type":"suggestion_type","suggestion_key":"suggestion_key" # you should copy the 'suggestion_key' but not the 'data_key'.}}]}}
    </task-requirements>

    """
    # with data_load_lock:
    if llm_name == 'gemini':
        print(video_path)
        video_file = llm_chat_client.files.upload(file=video_path)
        time.sleep(10)

        print(f"Completed upload: {video_file.uri}")
        while video_file.state.name == "PROCESSING":
            print('.', end='')
            time.sleep(1)
            video_file = llm_chat_client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(video_file.state.name)
        llm_response = llm_chat_client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[
                video_file,
                dataprompt,
                prompt
            ])

        print('\n\n', 'response:', llm_response)
        sugg = parse_llm_response(llm_response)
        print('sugg:', sugg)
        return sugg

    if llm_name == 'VideoChat-qwen2-7B':
        print(video_path)
        # time.sleep(10)
        prompt = prompt + dataprompt
        # region llm response
        url = "http://localhost:8111/chat/" # replace to the url of your VideoChat-qwen2-7B API endpoint
        payload = {
            "video_path": video_path,
            "user_prompt": prompt
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()
        llm_response = response.json()
        # endregion

        data=process_json_strings(llm_response)
        return data


def prompt_sensor_data(sround: RoundDataIncludesPoseSensor):
    prompt_unit = "Here is the sequence of the sensor data. We provide only the data of stroke moment, and the average value + Standard deviation around the peak \n"
    prompt_unit += "The input data is a sequence of sensor data, each with 16 attributes: \n"
    prompt_unit += "acc_x, acc_y, acc_z, agl_speed_x, agl_speed_y, agl_speed_z, agl_x, agl_y, agl_z, mgt_x, mgt_y, mgt_z, quat_1, quat_2, quat_3, quat_4\n"
    prompt_unit += "round_meta_info: " + sround.round_meta_info + "\n" + "stroke_number: " + str(
        len(sround.stroke_mask)) + "\n"
    stroke_len = np.sum(sround.stroke_mask)

    round_sensor_data = {}
    for idx, attr in enumerate(PREDEFINE_sensor_dim_keys):
        round_sensor_data[attr + '_stroke_moment'] = []
        round_sensor_data[attr + '_avg'] = []
        round_sensor_data[attr + '_std'] = []
    for stroke_sensor in sround.sensor_data[0:stroke_len]:
        for idx, attr in enumerate(PREDEFINE_sensor_dim_keys):
            round_sensor_data[attr + '_stroke_moment'].append(round(stroke_sensor[idx][50], 3))
            round_sensor_data[attr + '_avg'].append(round(np.mean(stroke_sensor[idx]), 3))
            round_sensor_data[attr + '_std'].append(round(np.std(stroke_sensor[idx]), 3))

    prompt_unit += json.dumps(round_sensor_data, indent=None)
    return prompt_unit


def prompt_pose_data(sround: RoundDataIncludesPoseSensor):
    prompt_unit = "Here is the sequence of the pose data. We provide only the data of stroke moment, and the average value + Standard deviation around the peak \n"
    prompt_unit += "The input data is a sequence of pose data, each with 3 attributes, describing the position of the joint: \n"
    prompt_unit += "x, y, z\n"
    prompt_unit += "round_meta_info: " + sround.round_meta_info + "\n" + "stroke_number: " + str(
        len(sround.stroke_mask)) + "\n"
    stroke_len = np.sum(sround.stroke_mask)

    round_pose_data = {}
    for idx, attr in enumerate(PREDEFINE_pose_dim_keys):
        round_pose_data[attr + '_stroke_moment'] = []
        round_pose_data[attr + '_avg'] = []
        round_pose_data[attr + '_std'] = []
    for stroke_pose in sround.pose_data[0:stroke_len]:
        for idx, attr in enumerate(PREDEFINE_pose_dim_keys):
            round_pose_data[attr + '_stroke_moment'].append(format_floats_in_list(stroke_pose[idx][5]))
            round_pose_data[attr + '_avg'].append(format_floats_in_list(np.mean(stroke_pose[idx])))
            round_pose_data[attr + '_std'].append(format_floats_in_list(np.std(stroke_pose[idx])))

    prompt_unit += json.dumps(round_pose_data, indent=None)
    return prompt_unit


def get_video_path(sround: RoundDataIncludesPoseSensor) -> str:
    if not hasattr(sround, 'video_path'):
        raise AttributeError(f"Object {type(sround).__name__} Lack: video_path")
    return sround.video_path


def generate_suggestion_struct(k):
    return {
        'content': "null",
        'suggestion_key': f'null{k}',
        'suggestion_type': f'null{k}'
    }


if __name__ == "__main__":
    sensor_input_size = len(PREDEFINE_sensor_dim_keys)
    fps = 10
    round_data_list: List[RoundDataIncludesPoseSensor] = load_round_data(sample_case_path)
    train_val_data, test_data = train_test_split(round_data_list, test_size=0.2, random_state=42)
    train_data, valid_data = train_test_split(train_val_data, test_size=0.25, random_state=42)
    all_data_prompt = []
    all_llm_outputs = []
    errcnt = 0
    # precision_list_key_types = []
    # precision_list_with_no_wrong_key = []
    # precision_list_key = []
    #
    # recall_list_key_type = []
    # recall_list_with_no_wrong_key = []
    # recall_list_key = []
    #
    # f1_list_key_type = []
    # f1_list_with_no_wrong_key = []
    # f1_list_key = []


    model_name = 'VideoChat-qwen2-7B'  # ['llama3.3-70b','gpt-4o','o1-mini' ,'claude-3-5-sonnet','claude-3-5-haiku','gemini' \
    # 'qwen2.5-72b-instruct','qwen2.5-32b-instruct','qwen-max','llama3.1-405b','llama3.1-70b','deepseek-v3']#[

    llm_client = load_llm_client(model_name)
    for round_idx in tqdm.tqdm(valid_data):
        round_ = valid_data[round_idx]
        all_sugg_types_from_exp: Set[
            Tuple[AllowedSuggestionKey, AllowedSuggestionType]] = set()  # ground truth key type pair
        for round_exp_sugs in round_.expert_suggestions:
            for stroke_exp_sug in round_exp_sugs.stroke_suggestions:
                sugg_type = stroke_exp_sug.suggestion.suggestion_type
                sugg_key = stroke_exp_sug.suggestion.key
                all_sugg_types_from_exp.add((sugg_key, sugg_type))
            for summary_exp_sug in round_exp_sugs.summary_suggestions:
                sugg_type = summary_exp_sug.suggestion_type
                sugg_key = summary_exp_sug.key
                all_sugg_types_from_exp.add((sugg_key, sugg_type))
        data_prompt = prompt_sensor_data(round_)
        all_data_prompt.append(data_prompt)
        video_path = get_video_path(round_)
        try:
            suggestions_by_llm = generate_sugg_one_round(data_prompt, llm_client, top_k, video_path,
                                                                        llm_name=model_name)
            print(suggestions_by_llm)
            all_llm_pred_sugg_key_types_this_round: Set[Tuple[str, str]] = set()  # predicted key type pair
            all_llm_pred_sugg_keys_from_llm_output = []  # predicted key
            # region correct empty suggestion
            k = 1
            while len(suggestions_by_llm['suggestions_list']) < top_k:
                suggestions_by_llm['suggestions_list'].append(generate_suggestion_struct(k))
                k = k + 1
            # endregion
            for sugg in suggestions_by_llm['suggestions_list']:
                all_llm_pred_sugg_key_types_this_round.add((sugg['suggestion_key'], sugg['suggestion_type']))
                all_llm_pred_sugg_keys_from_llm_output.append(sugg['suggestion_key'])

            all_sugg_key_from_exp = [tuple(sugg_key_type) for sugg_key_type in
                                     all_sugg_types_from_exp]  # ground truth key type pair tuple
            round_sugg_key_type_set = set(all_sugg_key_from_exp)  # ground truth key type tuple
            round_sugg_key_set = set(
                [sugg_key_type[0] for sugg_key_type in all_sugg_key_from_exp])  # ground truth key tuple

            print(all_llm_pred_sugg_key_types_this_round)
            print(round_sugg_key_type_set)

            all_llm_outputs.append(
                {"suggestions_by_llm": suggestions_by_llm['suggestions_list'],
                 'pred_pair_num': len(list(all_llm_pred_sugg_keys_from_llm_output)),
                 "all_llm_pred_sugg_key_types_this_round": list(all_llm_pred_sugg_key_types_this_round),
                 "true_set_num": len(list(round_sugg_key_type_set)), "true_set": list(round_sugg_key_type_set),
                 "true_sugg_keys_type": all_sugg_key_from_exp,
                 "round_.round_meta_info": round_.round_meta_info})
            time.sleep(1)
        except Exception as e:
            errcnt += 1
            print(f"Error: {e}")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    key_p, key_r, key_f1 = calc_key(all_llm_outputs)
    keytype_p, keytype_r, keytype_f1 = calc_keytype(all_llm_outputs)
    keytype_p_correctS1loss, keytype_r_correctS1loss, keytype_f1_correctS1loss = calc_keytype_after_correct_s1l(all_llm_outputs)
    print('avg_precision_key', round(key_p,5), '| avg_recall_key', round(key_r,5), '| avg_f1_key', round(key_f1,5))
    print('avg_precision_key_type', round(keytype_p,5), '| select_type', round(keytype_p_correctS1loss,5))
