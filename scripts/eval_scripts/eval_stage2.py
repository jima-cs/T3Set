# -*- coding: utf-8 -*-

"""
File Name: eval_stage2.py
Author: Ji Ma, Jiale Wu, Haoyu Wang
Date Created: 2025-02-11
Last Modified: 2025-02-26
Description: This script performs evaluation for stage 2 of our method. Its target is to evaluate the different LLM ability to select the suggestion Type.
WARNING: Please make sure to set the `FLAG_SENSECOACH=False` in `config.py`
"""

import tqdm
import json
import time
from typing import List, Tuple, Set
import re
from pydantic import BaseModel
import torch
import sys
import anthropic
from google import genai
import numpy as np
import os
from dotenv import load_dotenv

parent_dir = os.path.dirname(os.getcwd())
# model_dir = os.path.join(parent_dir, 'model')
# sys.path.append(model_dir)
load_dotenv('../../.env')

from data_loader_eval import load_round_data, sample_case_path, preprocess_data_as_data_loader
from new_key_mapping import allowed_sugg_keys, suggKey_to_dataKey_mapping, AllowedSuggestionKey, AllowedSuggestionType
from load_model import load_model
from key_dim import PREDEFINE_sensor_dim_keys, PREDEFINE_pose_dim_keys, RoundDataIncludesPoseSensor
from config import device, top_k, FLAG_SENSECOACH, trained_model_path
from openai import OpenAI
from typing import List

def load_llm_client(llm_name):
    if llm_name in ['deepseek-v3','deepseek-r1']:
        api_key = os.getenv("TENCENT_API_KEY")
        deepseek_client = OpenAI(
            api_key= api_key,
            base_url="https://api.lkeap.cloud.tencent.com/v1",
        )
        return deepseek_client
    elif llm_name in ['gpt-4o','o1-mini','o1-preview']:
        api_key = os.getenv("OPENAI_API_KEY")
        openai_client = OpenAI(
            api_key=api_key
        )
        return openai_client
    elif llm_name in['claude-3-5-sonnet','claude-3-5-haiku']:
        api_key = os.getenv("ANTHROPIC_API_KEY2")
        claude_client = anthropic.Anthropic(
            api_key= api_key,
        )
        return claude_client
    elif llm_name == 'gemini':
        api_key = os.getenv("GENAI_API_KEY")
        gemini_client = genai.Client(
            api_key=api_key, 
        )
        return gemini_client
    elif llm_name in ['qwen2.5-72b-instruct','qwen2.5-32b-instruct','qwen-max']:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        qwen_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        return qwen_client
    elif llm_name in ['llama3.1-405b','llama3.1-70b','llama3.3-70b']:
        api_key = os.getenv("LLAMA_API_KEY")
        llama_client = OpenAI(
            api_key = api_key,
            base_url = "https://api.llama-api.com"
        )
        return llama_client
    else:
        raise ValueError(f"Unsupported LLM: {llm_name}")



class SuggestionUnit(BaseModel):
    content: str
    suggestion_key: AllowedSuggestionKey
    suggestion_type: AllowedSuggestionType


class SuggestionsLLM(BaseModel):
    suggestions_list: List[SuggestionUnit]

system_prompt = f"""
    You are a table tennis coach. I have recorded video data and sensor data of a table tennis technique training session. 
    This session is called 'multi-ball' training, meaning the player strikes the ball using the same technique repeatedly for N times in a round.

    Your task is to analyze the suggestions I provide, each with a 'suggestion_key' and its related 'data_key' as well as the corresponding data of this 'data_key'. 
    I will give you M suggestion_key, you MUST response M suggestions according to the suggestion_key I gave you. 

    <data-description>
    Here is the explanation of the sensor data:
        - "acc_peak_exp_sqrt": The peak value of the acceleration in the x, y, z directions.
        - "agl_y_peak" : The peak value of the angle in the y direction.
        - "agl_spd_peak_exp_sqrt" : The peak value of the angular speed in the x, y, z directions.
    Here is the explanation of the pose data:
        The input data is a sequence of pose data, each with 3 attributes, describing the position of the joint
    </data-description>
    <task>
    Based on this information, you need to:
    
    1. Read the 'suggestion_key' and related data.
    2. Select a 'suggestion_type' from the predefined types.
    3. Generate a 'suggestion_content' sentence for each 'suggestion_key'. So, your suggestions number should equal to the number of input 'suggestion_key'.
    4. Output the suggestion in JSON format. ATTENTION! you should copy the 'suggestion_key' value but not the 'data_key'.

    </task>
    <task-requirements>
    The 'content' part should be a sentence describing the suggestion.
    The predefined types for'suggestion_type' are: ['center_of_gravity', 'right_wrist', 'right_shoulder', 'right_forearm', 'time_of_striking_ball',
                     'angle_of_racket', 'left_foot', 'right_foot', 'waist', 'right_upper_arm', 'left_shoulder',
                     'left_elbow', 'right_elbow', 'left_wrist', 'backswing_of_racket', 'left_hand', 'right_hand',
                     'left_leg', 'right_leg', 'left_knee', 'right_knee', 'right_finger', 'upper_body', 'grip_of_racket',
                     'others'], you should just copy the 'suggestion_key' from input but not change or create a new one.
    The predefined types for 'suggestion_type' are:
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
def generate_sugg_one_round(dataprompt, llm_chat_client,llm_name='gpt-4o'):
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
        # print(suggestion)
        return suggestion
    elif llm_name == 'o1-preview':
        completion = llm_chat_client.beta.chat.completions.parse(
            model="o1-preview",
            messages=[
                {'role': 'user', 'content':'system:content'+ prompt + 'user_content'+ dataprompt},
            ],
        )
        response = completion.choices[0].message.content
    elif llm_name == 'o1-preview':
        completion = llm_chat_client.beta.chat.completions.parse(
            model="o1-preview",
            messages=[
                {'role': 'user', 'content': 'system:content' + prompt + 'user_content' + dataprompt},
            ],
        )
        response = completion.choices[0].message.content
    elif llm_name == 'claude-3-5-sonnet':
        sg_claude = SuggestionsLLM.model_json_schema()
        completion = llm_chat_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            system = prompt,
            messages=[
                {'role': 'user', 'content': dataprompt},
            ],
        )
        response = completion.content[0].text
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
    elif llm_name == 'gemini':
        import enum
        class supported_type(enum.Enum):
            position = 'position'
            strength = 'strength'
            stability = 'stability'
            racket = 'racket'
            others = 'others'
        class supported_key(enum.Enum):
            center_of_gravity = 'center_of_gravity'
            right_wrist = 'right_wrist'
            right_shoulder = 'right_shoulder'
            right_forearm = 'right_forearm'
            time_of_striking_ball = 'time_of_striking_ball'
            angle_of_racket = 'angle_of_racket'
            left_foot = 'left_foot'
            right_foot = 'right_foot'
            waist = 'waist'
            right_upper_arm = 'right_upper_arm'
            left_shoulder = 'left_shoulder'
            left_elbow = 'left_elbow'
            right_elbow = 'right_elbow'
            left_wrist = 'left_wrist'
            backswing_of_racket = 'backswing_of_racket'
            left_hand = 'left_hand'
            right_hand = 'right_hand'
            left_leg = 'left_leg'
            right_leg = 'right_leg'
            left_knee = 'left_knee'
            right_knee = 'right_knee'
            right_finger = 'right_finger'
            upper_body = 'upper_body'
            grip_of_racket = 'grip_of_racket'
            others = 'others'
        class gemini_unit(BaseModel):
            content: str
            suggestion_key: supported_key
            suggestion_type: supported_type
        
        class gemini_suggestions(BaseModel):
            suggestions_list: List[gemini_unit]
        completion = llm_chat_client.generate_content(
            model='gemini-2.0-flash',
            
            contents=[dataprompt],
            config={
                'response_mime_type': 'application/json',
                'response_schema': gemini_suggestions,
                'system_instruction': prompt,
            },
        )
        suggestion = json.loads(completion.text)
        # print(suggestion)
        return suggestion
    elif llm_name == 'deepseek-v3':
        completion = llm_chat_client.beta.chat.completions.parse(
            model = "deepseek-v3",
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': dataprompt},
            ],
        )
        response = completion.choices[0].message.content      
    elif llm_name =='deepseek-r1':
        print('using deepseek r1')
        # time.sleep(1)
        try:
            completion = llm_chat_client.chat.completions.create(
                model="deepseek-r1",
                messages=[
                    {'role': 'user', 'content': prompt},
                    {'role': 'user', 'content': dataprompt},
                ],
                stream=True,
            )

            reasoning_content = ""
            content = ""
            for chunk in completion:
                if reasoning_content in chunk.choices[0].delta:
                    reasoning_content += chunk.choices[0].delta.reasoning_content
                elif chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            print(f"===== 模型推理过程 =====\n{reasoning_content}")
            print(f"===== 模型回复 =====\n{content}")
            response = content
        except Exception as e:
            print(f"Error: {e}")
            response = ""
    elif llm_name == 'qwen2.5-72b-instruct':
        completion = llm_chat_client.beta.chat.completions.parse(
            model="qwen2.5-72b-instruct",
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': dataprompt},
            ],
        )
        response = completion.choices[0].message.content
    elif llm_name == 'qwen2.5-32b-instruct':
        completion = llm_chat_client.beta.chat.completions.parse(
            model="qwen2.5-32b-instruct",
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': dataprompt},
            ],
        )
        response = completion.choices[0].message.content
    elif llm_name == 'qwen-max':
        completion = llm_chat_client.beta.chat.completions.parse(
            model="qwen-max-2024-09-19",
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': dataprompt},
            ],
        )
        response = completion.choices[0].message.content
    elif llm_name == 'llama3.1-70b':

        completion = llm_chat_client.beta.chat.completions.parse(
        model="llama3.1-70b",
        messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': dataprompt},
            ],
        )
        response = completion.choices[0].message.content
    elif llm_name == 'llama3.3-70b':

        completion = llm_chat_client.chat.completions.create(
        model="llama3.3-70b",
        messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': dataprompt},
            ],
        )
        response = completion.choices[0].message.content
    elif llm_name == 'llama3.1-405b':

        completion = llm_chat_client.beta.chat.completions.parse(
        model="llama3.1-405b",
        messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': dataprompt},
            ],
        )
        response = completion.choices[0].message.content
    else:
        raise ValueError(f"Unsupported LLM: {llm_name}")
    try:
        match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            json_content = match.group(1)
            suggestion = json.loads(json_content)
            # print(suggestion)
            return suggestion
        else:
            raise ValueError("Failed to find JSON marked content in the response")
    except json.JSONDecodeError:
        raise ValueError("Failed to parse LLM response as JSON")



def evaluate_suggestion_f1(predicted_key_type: Set[Tuple[str, str]], true_key_type: Set[Tuple[str, str]]):
    """
    F1 score of suggestion_type

    :param predicted_key_type: pred suggestion_key_type
    :param true_key_type: true suggestion_key_type
    :return: F1 score
    """
    predicted_set = predicted_key_type
    true_set = true_key_type
    precision = len(predicted_set & true_set) / len(predicted_set) if len(predicted_set) > 0 else 0
    recall = len(predicted_set & true_set) / len(true_set) if len(true_set) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1



#calculate the precision, recall, f1 of key_type after correct the Stage 1 Loss
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
        if(len(sugg_key_set) != 6):
            print(round_['round_.round_meta_info'])
            print(sugg_key_set)            
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



class RoundDataWithSuggKeys(BaseModel):
    pred_sugg_keys: List[AllowedSuggestionKey]
    true_sugg_keys: List[AllowedSuggestionKey]
    sugg_key_type_set: Set[Tuple[str, str]]
    round_meta_info: str
    pose_data: List[List[List[List[float]]]]
    sensor_data: List[List[List[float]]]
    stroke_mask: List[int]


def get_sugg_keys(model, test_data_loader, topk=6) -> List[RoundDataWithSuggKeys]:
    """
    get topk suggestions keys from the model output
    :param model: trained model
    :param test_data_loader: load the ground truth and origin dataset
    :param topk: topK
    :return: structured suggestion list
    """
    round_data_with_gen_sugg_keys = []
    with torch.no_grad():
        for batch in test_data_loader:
            sensor_x, pose_x, stroke_mask, round_meta_info, sugg_key_type_set, targets = batch
            sensor_x = sensor_x.to(device, non_blocking=True)
            pose_x = pose_x.to(device, non_blocking=True)
            stroke_mask = stroke_mask.to(device, non_blocking=True)
            logits = model(sensor_x, pose_x, stroke_mask)
            probs = torch.sigmoid(logits)  # need when inference, no need for training
            _, topk_indices = probs.topk(topk, dim=1)  # [N, k]
            for batch_idx, batch_sugg_keys in enumerate(topk_indices.tolist()):
                true_keys = []
                for (key,type) in sugg_key_type_set[batch_idx]:
                    true_keys.append(key)
                round_data_with_gen_sugg_keys.append(RoundDataWithSuggKeys(
                    pred_sugg_keys=[allowed_sugg_keys[idx] for idx in batch_sugg_keys],
                    true_sugg_keys=true_keys,
                    round_meta_info=round_meta_info[batch_idx],
                    pose_data=pose_x[batch_idx].tolist(),
                    sensor_data=sensor_x[batch_idx].tolist(),
                    stroke_mask=stroke_mask[batch_idx].tolist(),
                    sugg_key_type_set=sugg_key_type_set[batch_idx]
                ))

    return round_data_with_gen_sugg_keys


def format_floats_in_list(item):
    if isinstance(item, float):
        return f"{item:.3f}"
    elif isinstance(item, list):
        return [format_floats_in_list(subitem) for subitem in item]
    else:
        return item


def merge_data_according_to_sugg_keys(sround: RoundDataWithSuggKeys):
    prompt_unit = ""
    prompt_unit += "round_meta_info: " + sround.round_meta_info + "\n" + "stroke_number: " + str(
        len(sround.stroke_mask)) + "\n"
    related_data_key_set = set()
    keys_give_into_llm = []
    for sugg_key in sround.pred_sugg_keys:

        keys_give_into_llm.append(sugg_key)
        stroke_unit_string = ""
        stroke_unit_string += "suggestion key: " + sugg_key + "\n"
        stroke_unit_string += "related data keys: " + str(suggKey_to_dataKey_mapping[sugg_key]) + "\n"
        for data_key in suggKey_to_dataKey_mapping[sugg_key]:
            related_data_key_set.add(data_key)
    prompt_unit += f"You are expected to generate {len(keys_give_into_llm)} suggestions according to the the {len(keys_give_into_llm)} suggestion_keys: {keys_give_into_llm}.\n"
    prompt_unit += "We retrievaled the following data of this round for each data_key. Here is the sequence of the pose data.\
                     We provide only the data of stroke moment, and the average value + Standard deviation around the peak"
    for data_key in related_data_key_set:
        stroke_len = np.sum(sround.stroke_mask)
        if data_key in PREDEFINE_sensor_dim_keys:
            round_sensor_data = [stroke[PREDEFINE_sensor_dim_keys.index(data_key)] for stroke in
                                 sround.sensor_data[0:stroke_len]]
            prompt_unit += "- " + data_key + str(format_floats_in_list(round_sensor_data[50])) + "\n" # sensor peak
            prompt_unit += "- " + data_key + "_avg: " +  str(format_floats_in_list(np.mean(round_sensor_data)))+ "\n"
            prompt_unit += "- " + data_key + "_std: " + str(format_floats_in_list(np.std(round_sensor_data)))+ "\n"
        elif data_key in PREDEFINE_pose_dim_keys:
            round_pose_data = [stroke[PREDEFINE_pose_dim_keys.index(data_key)] for stroke in
                               sround.pose_data[0:stroke_len]]
            peak_pose = [stroke[5] for stroke in round_pose_data] 
            prompt_unit += "- " + data_key + str(format_floats_in_list(peak_pose)) + "\n"
            prompt_unit += "- " + data_key + "_avg: " + str(format_floats_in_list(np.mean(round_pose_data))) + "\n"
            prompt_unit += "- " + data_key + "_std: " + str(format_floats_in_list(np.std(round_pose_data))) + "\n"

    return prompt_unit

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    else:
        return obj


if __name__ == "__main__":
    sensor_input_size = len(PREDEFINE_sensor_dim_keys)
    fps = 10
    use_lstm = False
    pose_input_size = fps * len(PREDEFINE_pose_dim_keys) * 3 
    num_keys = len(PREDEFINE_sensor_dim_keys) + len(PREDEFINE_pose_dim_keys)
    local_model_path = trained_model_path
    model = load_model(sensor_input_size, pose_input_size, USE_LSTM_LAYER=use_lstm, path=local_model_path, USE_SENSOR_PROCESS=FLAG_SENSECOACH)
    batch_size = 5
    round_data_list: List[RoundDataIncludesPoseSensor] = load_round_data(sample_case_path)
    train_set, valid_set, test_set = preprocess_data_as_data_loader(round_data_list, batch_size, fps)
    
    # valid set和test set都是没有被用于训练的
    round_data_with_gen_sugg_keys = get_sugg_keys(model, valid_set, topk=top_k)
    errcnt = 0
    all_llm_outputs = []
    precision_list = []
    recall_list = []
    f1_list = []
    model_name = 'gpt-4o'
    llm_client = load_llm_client(model_name)
    for round_ in tqdm.tqdm(round_data_with_gen_sugg_keys):
        merged_data_string = merge_data_according_to_sugg_keys(round_)
        try:

            suggestions_by_llm = generate_sugg_one_round(merged_data_string, llm_client, llm_name = model_name)
            all_llm_pred_sugg_key_types_this_round: Set[Tuple[str, str]] = set()
            all_llm_pred_sugg_keys_from_llm_output = []
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
            all_model_pred_sugg_keys = round_.pred_sugg_keys

            for sugg in suggestions_by_llm['suggestions_list']:
                all_llm_pred_sugg_key_types_this_round.add((sugg['suggestion_key'], sugg['suggestion_type']))
                all_llm_pred_sugg_keys_from_llm_output.append(sugg['suggestion_key'])
            true_key_type_set = set()
            round_.sugg_key_type_set = [tuple(sugg_key) for sugg_key in round_.sugg_key_type_set]
            round_sugg_key_type_set = set(round_.sugg_key_type_set)
            for key_type in round_sugg_key_type_set:
                true_key_type_set.add(key_type)
            precesion, recall, f1 = evaluate_suggestion_f1(all_llm_pred_sugg_key_types_this_round, true_key_type_set)
            pred_key_in_true_list = []
            for pred_k in round_.pred_sugg_keys:
                if pred_k in round_.true_sugg_keys:
                    pred_key_in_true_list.append(pred_k)
            all_llm_outputs.append(
                {"suggestions_by_llm": suggestions_by_llm['suggestions_list'], 'pred_pair_num': len(list(all_llm_pred_sugg_keys_from_llm_output)),
                "all_llm_pred_sugg_key_types_this_round": list(all_llm_pred_sugg_key_types_this_round),
                "true_set_num": len(list(true_key_type_set)), "true_set": list(true_key_type_set),
                "precision": round(precesion, 4), "recall": round(recall, 4), "f1": round(f1),
                "round_.round_meta_info": round_.round_meta_info, "keys_give_to_llm": round_.pred_sugg_keys,"ruond_.true_sugg_keys": round_.true_sugg_keys,
                "round_.sugg_key_type_set": round_.sugg_key_type_set})
            time.sleep(1)
        except Exception as e:
            errcnt += 1
            print(e)
            
    key_p, key_r, key_f1 = calc_key(all_llm_outputs)
    keytype_p, keytype_r, keytype_f1 = calc_keytype(all_llm_outputs)
    keytype_p_correctS1loss, keytype_r_correctS1loss, keytype_f1_correctS1loss = calc_keytype_after_correct_s1l(all_llm_outputs)
    print('avg_precision_key_type', round(keytype_p,5), '| select_type', round(keytype_p_correctS1loss,5))