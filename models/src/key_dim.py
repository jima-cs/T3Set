import re
from typing import Tuple, Optional, List, Union, Literal, Set, Dict, get_args
from pydantic import BaseModel
from setting import USE_SENSOR_PROCESS
from new_key_mapping import AllowedSuggestionType, AllowedSuggestionKey

sensor_phy_dim_keys = ["acc_x", "acc_y", "acc_z", "agl_speed_x", "agl_speed_y", "agl_speed_z", "agl_x", "agl_y", "agl_z",
                "mgt_x", "mgt_y", "mgt_z", "quat_1", "quat_2",
                "quat_3", "quat_4"]

PREDEFINE_sensor_dim_keys = ["acc_peak_exp_sqrt", "agl_y_peak", "agl_spd_peak_exp_sqrt"] if USE_SENSOR_PROCESS else sensor_phy_dim_keys
pose_keypoints_with_official_name = [
    {"id": 0, "name": "Nose"},  
    {"id": 1, "name": "Left eye inner"},  
    {"id": 2, "name": "Left eye"},  
    {"id": 3, "name": "Left eye outer"},  
    {"id": 4, "name": "Right eye inner"},  
    {"id": 5, "name": "Right eye"},  
    {"id": 6, "name": "Right eye outer"},  
    {"id": 7, "name": "Left ear"},  
    {"id": 8, "name": "Right ear"},  
    {"id": 9, "name": "Mouth left"},  
    {"id": 10, "name": "Mouth right"},  
    {"id": 11, "name": "Left shoulder"},  
    {"id": 12, "name": "Right shoulder"},  
    {"id": 13, "name": "Left elbow"},  
    {"id": 14, "name": "Right elbow"},  
    {"id": 15, "name": "Left wrist"},  
    {"id": 16, "name": "Right wrist"},  
    {"id": 17, "name": "Left pinky"},  
    {"id": 18, "name": "Right pinky"},  
    {"id": 19, "name": "Left index"},  
    {"id": 20, "name": "Right index"},  
    {"id": 21, "name": "Left thumb"},  
    {"id": 22, "name": "Right thumb"},  
    {"id": 23, "name": "Left hip"},  
    {"id": 24, "name": "Right hip"},  
    {"id": 25, "name": "Left knee"},  
    {"id": 26, "name": "Right knee"},  
    {"id": 27, "name": "Left ankle"},  
    {"id": 28, "name": "Right ankle"},  
    {"id": 29, "name": "Left heel"},  
    {"id": 30, "name": "Right heel"},  
    {"id": 31, "name": "Left foot index"},  
    {"id": 32, "name": "Right foot index"}  
]
PREDEFINE_pose_dim_keys = [point["name"].lower().replace(' ', '_') for point in pose_keypoints_with_official_name]
all_data_keys = PREDEFINE_pose_dim_keys + PREDEFINE_sensor_dim_keys 



def validate_and_convert_suggkey(stroke_index, key:str, suggestion_type: str):
    try:
        stroke_index = int(stroke_index)
    except ValueError:
        print("Invalid stroke index; defaulting to 0")
        stroke_index = 0  # Default to 0 if conversion fails

    allowed_types = get_args(AllowedSuggestionType)
    if suggestion_type not in allowed_types:
        suggestion_type = 'others'

    allowed_keys = get_args(AllowedSuggestionKey)
    if key not in allowed_keys:
        key = 'others'

    return stroke_index, key, suggestion_type
class ParsedSuggestion(BaseModel):
    context: str
    key: AllowedSuggestionKey
    suggestion_type: AllowedSuggestionType


class StrokeExpertSuggestion(BaseModel):
    stroke_index: int
    suggestion: ParsedSuggestion


class RoundExpertSuggestions(BaseModel):
    summary_suggestions: List[ParsedSuggestion]
    stroke_suggestions: List[StrokeExpertSuggestion]
    suggestion_keys: List[AllowedSuggestionKey]

class RoundDataIncludesPoseSensor(BaseModel):
    round_meta_info: str
    pose_data: List[List[List[List[float]]]]
    sensor_data: List[List[List[float]]]
    stroke_mask: List[int]
    expert_suggestions: List[RoundExpertSuggestions]
    expert_sugg_key_id_set: Set[int]




