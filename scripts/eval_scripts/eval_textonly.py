# -*- coding: utf-8 -*-

"""
File Name: eval_textonly.py
Author: Ji Ma
Date Created: 2025-02-11
Last Modified: 2025-02-26
Description: This script performs evaluation for text LLM of our baseline. Its target is to evaluate the different LLM ability to select the suggestion key, select the suggestion (key,type), generating content.
WARNING: Please make sure to set the `FLAG_SENSECOACH=False` in `config.py`
"""
import os
import json
import re
import sys
import numpy as np
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from typing import List, Union, Literal, Tuple, Set
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
import anthropic

from config import top_k

# parent_dir = os.path.dirname(os.getcwd())
# model_dir = os.path.join(parent_dir, 'model')
# sys.path.append(model_dir)
load_dotenv('../../.env')
from data_loader_eval import load_round_data, sample_case_path
from key_dim import PREDEFINE_sensor_dim_keys, PREDEFINE_pose_dim_keys, RoundDataIncludesPoseSensor

AllowedSuggestionType = Union[Literal['position'], Literal['strength'], Literal['stability'], Literal['racket'],
Literal['others']]

AllowedSuggestionKey = Union[Literal['center_of_gravity'],
Literal["right_wrist"], Literal["right_shoulder"],
Literal["right_forearm"], Literal['time_of_striking_ball'],
Literal["angle_of_racket"], Literal['left_foot'], Literal['right_foot'], Literal['waist'],
Literal["right_upper_arm"],
Literal["left_shoulder"], Literal["left_elbow"], Literal["right_elbow"],
Literal["left_wrist"],
Literal["backswing_of_racket"], Literal["left_hand"], Literal["right_hand"],
Literal["left_leg"], Literal["right_leg"], Literal["left_knee"], Literal["right_knee"],
Literal["right_finger"], Literal["upper_body"],
Literal['grip_of_racket'], Literal['others']]


class SuggestionUnit(BaseModel):
    content: str
    suggestion_key: AllowedSuggestionKey
    suggestion_type: AllowedSuggestionType


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
    if llm_name in ['gpt-4o', 'o1-preview']:
        api_key = os.getenv("OPENAI_API_KEY")
        openai_client = OpenAI(
            api_key=api_key
        )
        return openai_client
    elif llm_name == 'gemini':
        api_key = os.getenv("GENAI_API_KEY")
        gemini_client = genai.Client(
            api_key=api_key,
        )
        return gemini_client
    elif llm_name in ['deepseek-v3','deepseek-r1']:
        api_key = os.getenv("TENCENT_API_KEY")
        deepseek_client = OpenAI(
            api_key= api_key,
            base_url="https://api.lkeap.cloud.tencent.com/v1",
        )
        return deepseek_client
    elif llm_name in['claude-3-5-sonnet','claude-3-5-haiku']:
        api_key = os.getenv("ANTHROPIC_API_KEY3")
        claude_client = anthropic.Anthropic(
            api_key= api_key,
        )
        return claude_client
    elif llm_name in ['llama3.1-405b','llama3.1-70b','llama3.3-70b']:
        api_key = os.getenv("LLAMA_API_KEY")
        llama_client = OpenAI(
            api_key = api_key,
            base_url = "https://api.llama-api.com"
        )
        return llama_client

topk = top_k
system_prompt =f"""
    You are a table tennis coach. I have recorded video data and sensor data of a table tennis technique training session. 
    This session is called 'multi-ball' training, meaning the player strikes the ball using the same technique repeatedly for N times in a round.

    Your task is to analyze the data I provide, each with a sequence of pose data and the corresponding sensor data, 
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

    <output format>```json
    {{
        "suggestions_list": [
            {{
                "content": "suggestion content",
                "suggestion_type": "suggestion_type",
                "suggestion_key": "suggestion_key" # you should copy the 'suggestion_key' but not the 'data_key'.
            }}
        ]
    }}```
    </output format>
    </task-requirements>
    """
def generate_sugg_one_round(dataprompt, llm_chat_client, topk, llm_name='gpt-4o'):
    prompt = system_prompt

    if llm_name == 'gpt-4o':
        completion = llm_chat_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {'role': 'developer', 'content': prompt},
                {'role': 'user', 'content': dataprompt},
            ],
            response_format=SuggestionsLLM
        )
        suggestion = json.loads(completion.choices[0].message.content)
        return suggestion
    elif llm_name == 'o1-preview':
        completion = llm_chat_client.beta.chat.completions.parse(
            model="o1-preview",
            messages=[
                {'role': 'user', 'content': 'system:content' + prompt + 'user_content' + dataprompt},
            ],
        )
        response = completion.choices[0].message.content
    elif llm_name == 'claude-3-5-haiku':
        sg_claude = SuggestionsLLM.model_json_schema()
        completion = llm_chat_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=8192,
            system = prompt,
            messages=[
                {'role': 'user', 'content': dataprompt},
            ],
        )
        response = completion.content[0].text
    elif llm_name == 'llama3.3-70b':
        completion = llm_chat_client.chat.completions.create(
        model="llama3.3-70b",
        messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': dataprompt},
            ],
        )
        response = completion.choices[0].message.content
    elif llm_name =='deepseek-r1':
        try:
            completion = llm_chat_client.chat.completions.create(
                model="deepseek-r1",
                messages=[
                    {'role': 'user', 'content': prompt},
                    {'role': 'user', 'content': dataprompt},
                ],
                stream=True,
                temperature=0.6,
            )

            reasoning_content = ""
            content = ""
            for chunk in completion:
                if reasoning_content in chunk.choices[0].delta:
                    reasoning_content += chunk.choices[0].delta.reasoning_content
                elif chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            response = content
        except Exception as e:
            print(f"Error: {e}")
            response = ""
    else:
        raise ValueError(f"Unsupported LLM: {llm_name}")
    try:
        match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            json_content = match.group(1)
            suggestion = json.loads(json_content)
            return suggestion
        else:
            raise ValueError("Failed to find JSON marked content in the response")
    except json.JSONDecodeError:
        raise ValueError("Failed to parse LLM response as JSON")


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
    # print(prompt_unit)
    return prompt_unit

import tqdm
def check_round(llm_record:str):
    with open(llm_record,"r") as f:
        llm_record_data = json.load(f)
    round_data_list: List[RoundDataIncludesPoseSensor] = load_round_data(sample_case_path)
    train_val_data, test_data = train_test_split(round_data_list, test_size=0.2, random_state=42)
    train_data, valid_data = train_test_split(train_val_data, test_size=0.25, random_state=42)
    meta_data_list = []
    for round_ in llm_record_data['output']:
        meta_data_list.append(round_['round_.round_meta_info'])
    round_data = [data.round_meta_info for data in valid_data if data.round_meta_info not in meta_data_list]

    return round_data

def evaluate_suggestion_f1(predicted_key_type: Set[Tuple[str, str]], true_key_type: Set[Tuple[str, str]]):
    predicted_set = predicted_key_type
    true_set = true_key_type
    precision = len(predicted_set & true_set) / len(predicted_set) if len(predicted_set) > 0 else 0
    recall = len(predicted_set & true_set) / len(true_set) if len(true_set) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

#calculate the precision, recall, f1 of key_type after removing the wrong key
def calc_keytype_after_correct_s1l(llm_record_output):

    precision_list_with_no_wrong_key = []
    recall_list_with_no_wrong_key = []
    f1_list_with_no_wrong_key = []
    for round_ in llm_record_output:
        all_llm_pred_sugg_key_types_this_round = [tuple(item) for item in round_['all_llm_pred_sugg_key_types_this_round']]
        round_sugg_key_set = set([item[0] for item in round_['true_set']])#set(round_['ruond_.true_sugg_keys'])
        round_sugg_key_type_set = set([tuple(item) for item in round_['true_set']])

        filtered_sugg_key_type_set = set(
                [sugg_key_type for sugg_key_type in all_llm_pred_sugg_key_types_this_round if
                    sugg_key_type[0] in round_sugg_key_set])
        precission_key_type_no_wrong_key, recall_key_type_no_wrong_key, f1_key_type_no_wrong_key = \
                evaluate_suggestion_f1(filtered_sugg_key_type_set, round_sugg_key_type_set)
        precision_list_with_no_wrong_key.append(precission_key_type_no_wrong_key)
        recall_list_with_no_wrong_key.append(recall_key_type_no_wrong_key)
        f1_list_with_no_wrong_key.append(f1_key_type_no_wrong_key)
    avg_precision_key_type_no_wrong_key = sum(precision_list_with_no_wrong_key)/len(precision_list_with_no_wrong_key)
    avg_recall_key_type_no_wrong_key = sum(recall_list_with_no_wrong_key)/len(recall_list_with_no_wrong_key)
    avg_f1_key_type_no_wrong_key = sum(f1_list_with_no_wrong_key)/len(f1_list_with_no_wrong_key)
    print("avg_precision_key_type_no_wrong_key:",avg_precision_key_type_no_wrong_key,"avg_recall_key_type_no_wrong_key:",avg_recall_key_type_no_wrong_key,"avg_f1_key_type_no_wrong_key:",avg_f1_key_type_no_wrong_key)    
    return avg_precision_key_type_no_wrong_key,avg_recall_key_type_no_wrong_key,avg_f1_key_type_no_wrong_key

#calculate the precision, recall, f1 of key
def calc_key(llm_record_output):
    precision_list_key = []
    recall_list_key = []
    f1_list_key = []
    for round_ in llm_record_output:
        round_sugg_key_set = set([item[0] for item in round_['true_set']])#
        sugg_key_set = set(
                [sugg_key[0] for sugg_key in round_['all_llm_pred_sugg_key_types_this_round']])
        precission_key, recall_key, f1_key = \
                evaluate_suggestion_f1(sugg_key_set, round_sugg_key_set)
        precision_list_key.append(precission_key)
        recall_list_key.append(recall_key)
        f1_list_key.append(f1_key)
    avg_precision_key = sum(precision_list_key)/len(precision_list_key)
    avg_recall_key = sum(recall_list_key)/len(recall_list_key)
    avg_f1_key = sum(f1_list_key)/len(f1_list_key)
    print("avg_precision_key:",avg_precision_key,"avg_recall_key:",avg_recall_key,"avg_f1_key:",avg_f1_key)
    return avg_precision_key,avg_recall_key,avg_f1_key

def calc_keytype(llm_record_output):
    precision_list_with_no_wrong_key = []
    recall_list_with_no_wrong_key = []
    f1_list_with_no_wrong_key = []
    for round_ in llm_record_output:
        all_llm_pred_sugg_key_types_this_round = set([tuple(item) for item in round_['all_llm_pred_sugg_key_types_this_round']])
        round_sugg_key_type_set = set([tuple(item) for item in round_['true_set']])
        precission_key_type_no_wrong_key, recall_key_type_no_wrong_key, f1_key_type_no_wrong_key = \
                evaluate_suggestion_f1(all_llm_pred_sugg_key_types_this_round, round_sugg_key_type_set)
        precision_list_with_no_wrong_key.append(precission_key_type_no_wrong_key)
        recall_list_with_no_wrong_key.append(recall_key_type_no_wrong_key)
        f1_list_with_no_wrong_key.append(f1_key_type_no_wrong_key)
    avg_precision_key_type_no_wrong_key = sum(precision_list_with_no_wrong_key)/len(precision_list_with_no_wrong_key)
    avg_recall_key_type_no_wrong_key = sum(recall_list_with_no_wrong_key)/len(recall_list_with_no_wrong_key)
    avg_f1_key_type_no_wrong_key = sum(f1_list_with_no_wrong_key)/len(f1_list_with_no_wrong_key)
    print("avg_precision_key_type:",avg_precision_key_type_no_wrong_key,"avg_recall_key_type:",avg_recall_key_type_no_wrong_key,"avg_f1_key_type:",avg_f1_key_type_no_wrong_key)    
    return avg_precision_key_type_no_wrong_key,avg_recall_key_type_no_wrong_key,avg_f1_key_type_no_wrong_key

if __name__ == "__main__":
    sensor_input_size = len(PREDEFINE_sensor_dim_keys)
    fps = 10
    
    pose_input_size = fps * len(PREDEFINE_pose_dim_keys) * 3  # 3是xyz
    num_keys = len(PREDEFINE_sensor_dim_keys) + len(PREDEFINE_pose_dim_keys)
    batch_size = 5
    round_data_list: List[RoundDataIncludesPoseSensor] = load_round_data(sample_case_path)
    train_val_data, test_data = train_test_split(round_data_list, test_size=0.2, random_state=42)
    train_data, valid_data = train_test_split(train_val_data, test_size=0.25, random_state=42)
    
    errcnt = 0
    all_llm_outputs = []
    # model_list = ['o1-preview']  # ['llama3.3-70b','gpt-4o','o1-mini' ,'claude-3-5-sonnet','claude-3-5-haiku','gemini' \
    # 'qwen2.5-72b-instruct','qwen2.5-32b-instruct','qwen-max','llama3.1-405b','llama3.1-70b','deepseek-v3']#[
    model_name = 'gpt-4o'
    llm_client = load_llm_client(model_name)
    # llm_client = load_llm_client(model_name)
    for round_ in tqdm.tqdm(valid_data):
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
        data_prompt = prompt_pose_data(round_) + prompt_sensor_data(round_)
        try:
            suggestions_by_llm = generate_sugg_one_round(data_prompt, llm_client, top_k, llm_name=model_name)
            all_llm_pred_sugg_key_types_this_round: Set[Tuple[str, str]] = set()  # predicted key type pair
            all_llm_pred_sugg_keys_from_llm_output = []  # predicted key
            # region correct empty suggestion
            k = 1
            while len(suggestions_by_llm['suggestions_list']) < top_k:
                suggestions_by_llm['suggestions_list'].append({
                    'content': "null",
                    'suggestion_key': f'null{k}',
                    'suggestion_type': f'null{k}'
                })
                k = k + 1
            # endregion
            print('generated ', len(suggestions_by_llm['suggestions_list']),' topk', top_k)
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
    
    key_p, key_r, key_f1 = calc_key(all_llm_outputs)
    keytype_p, keytype_r, keytype_f1 = calc_keytype(all_llm_outputs)
    keytype_p_correctS1loss, keytype_r_correctS1loss, keytype_f1_correctS1loss = calc_keytype_after_correct_s1l(all_llm_outputs)
    print('avg_precision_key', round(key_p,5), '| avg_recall_key', round(key_r,5), '| avg_f1_key', round(key_f1,5))
    print('avg_precision_key_type', round(keytype_p,5), '| select_type', round(keytype_p_correctS1loss,5))
