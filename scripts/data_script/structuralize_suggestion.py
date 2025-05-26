from typing import Union, Literal

from openai import OpenAI, AzureOpenAI
import json
import os
import sys
from pydantic import BaseModel
from dotenv import load_dotenv
sys.path.append(os.getcwd())
load_dotenv('../../.env')
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
class Comment(BaseModel):
    start: str
    end: str
    text: str
    comment_zh: str
    comment_en: str
    suggestion_key: AllowedSuggestionKey
    suggestion_type: AllowedSuggestionType
    index: int
    stroke_index: int
    type: str


class Comments(BaseModel):
    comments: list[Comment]


comments_json_files = []

def text_summarize(client, root_path):
    comments_json_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                parts = file_path.split(os.sep)
                if len(parts) >= 3 and parts[-3] == "comments" and parts[-2] in [
                    "realtime", "global"]:
                    comments_json_files.append(file_path)

    for test_file in comments_json_files:
        target_folder = os.path.dirname(test_file) + "_sugg_key"
        print('target_folder :', target_folder, test_file)
        with open(test_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        json_string = json.dumps(data)
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": "<task>You are a table tennis coach using English, now here is a json file containing some advices commenting a video, \
                            which consists of a sequence of {\"start\", \"end\", \"text\", \"index\",\"stroke_index\",\"type\"}. You should do the tasks below one by one. \
                            1. Translate the \"text\" field into English, Add a new field 'comment_en', output your translation.\
                            2. Find a Suggestion Object this sentence comments on, add a new field 'suggestion_key', store the object you choose, You MUST only use the SuggestionKey in the list below:\
                            AllowedSuggestionKey = Union[\
                                Literal[\"center_of_gravity\"], Literal[\"right_wrist\"], Literal[\"right_shoulder\"],\
                                Literal[\"right_forearm\"], Literal[\"time_of_striking_ball\"], Literal[\"angle_of_racket\"],\
                                Literal[\"left_foot\"], Literal[\"right_foot\"], Literal[\"waist\"], Literal[\"right_upper_arm\"],\
                                Literal[\"left_shoulder\"], Literal[\"left_elbow\"], Literal[\"right_elbow\"], Literal[\"left_wrist\"],\
                                Literal[\"backswing of racket\"], Literal[\"left_hand\"], Literal[\"right_hand\"], Literal[\"left_leg\"],\
                                Literal[\"right_leg\"], Literal[\"left_knee\"], Literal[\"right_knee\"], Literal[\"right_finger\"],\
                                Literal[\"upper_body\"], Literal[\"left_foot_index\"], Literal[\"right_foot_index\"],\
                                Literal[\"grip of racket\"], Literal[\"others\"]\
                            ]. You MUST follow the rules in '<addition>' part below \
                            3. Conclude what type of the suggestions of the sentence is, add a new field 'suggestion_type', store the type you choose. You MUST only use five values below:\
                            position: Indicates that the position of this key is incorrect.\
                            strength: Indicates that the force exertion or explosiveness of this key is incorrect or insufficient.\
                            stability: Indicates that the stability of this key is incorrect.\
                            racket: Indicates that the relationship between this key and the racket is incorrect.\
                            others: Represents other types of suggestions related to this key.Don't classify them as `others` unless you really need to.\
                            Each record must only comment on one suggestion key.\
                            4. copy the content of 'index' into a new field 'comment_index', and delete the original 'index' field.\
                            </task>\
                            <addition>\
                            1. If the 'start' of this record is similar to the 'end' of the last one and as you think the two records describe the same action type of the same object,\
                            you need to concatenate the records as one record, and Find the objects the sentence comments on. IF THE CONCATENETED RECORD COMMENTS ON\
                                MULTIPLE OBJECTS, FOLLOW THE SECOND POINT OF ADDITION AND FIND ALL SUGGESTION OBJECTS WITH NO IGNORANCE\n\
                            2. When multiple parts are involved in one record, it is allowed to summarize a Chinese record into multiple records and increase the index, \
                                but 'start' and 'end' should remain the same as the original comment.\n\
                            3. The text input sometimes comments the wrong motion, though it might be similar to an Imperative Sentence, you need to \
                                complete the comment as 'Shouldn't do' \n\
                            </addition>\
                            <example1>Here is an example, input:  [{\
                                    \"start\": \"00:01:04,260\",\
                                    \"end\": \"00:01:08,019\",\
                                    \"text\": \"然后小臂稍微收小一点多用手腕的力量\",\
                                    \"index\": 12,\
                                    \"stroke_index\": 12,\
                                    \"type\": \"global\"\
                                }]\
                                output:[{\
                                    \"start\": \"00:01:04,260\",\
                                    \"end\": \"00:01:08,019\",\
                                    \"text\": \"然后小臂稍微收小一点多用手腕的力量\",\
                                    \"comment_en\": \"right forearm should retract slightly\",\
                                    \"suggestion_key\": \"right_forearm\",\
                                    \"suggestion_type\": \"position\",\
                                    \"comment_index\": 12,\
                                    \"stroke_index\": 12,\
                                    \"type\": \"global\",\
                                },\
                                {\
                                    \"start\": \"00:01:04,260\",\
                                    \"end\": \"00:01:08,019\",\
                                    \"text\": \"然后小臂稍微收小一点多用手腕的力量\",\
                                    \"comment_en\": \"right wrist should use more wrist power\",\
                                    \"suggestion_key\": \"right_wrist\",\
                                    \"suggestion_type\": \"strength\",\
                                    \"comment_index\": 13,\
                                    \"stroke_index\": 12,\
                                    \"type\": \"global\"\
                                }]\
                            </example1>\
                            <example2>Here is another example, input: [{\
                                    \"start\": \"00:01:16,019\",\
                                    \"end\": \"00:01:19,760\",\
                                    \"text\": \"然后重心稍微身体稍微向前轻一点\",\
                                    \"index\": 18,\
                                    \"stroke_index\": 12,\
                                    \"type\": \"global\"\
                                }]\
                                output:[{\
                                    \"start\": \"00:01:16,019\",\
                                    \"end\": \"00:01:19,760\",\
                                    \"text\": \"然后重心稍微身体稍微向前倾一点\",\
                                    \"comment_en\": \"center of gravity should not lean backward, slightly lean forward\",\
                                    \"suggestion_key\": \"center_of_gravity\",\
                                    \"suggestion_type\": \"position\",\
                                    \"comment_index\": 18,\
                                    \"stroke_index\": 12,\
                                    \"type\": \"global\"\
                                }]\
                            </example2>\
                            <example3>Here is another example, input:  [{\
                                \"start\": \"00:00:23,040\",\
                                \"end\": \"00:00:25,540\",\
                                \"text\": \"多用一点手腕的力量\",\
                                \"index\": 2\
                                \"stroke_index\": 12,\
                                }]\
                                output:[{\
                                \"start\": \"00:00:23,040\",\
                                \"end\": \"00:00:25,540\",\
                                \"text\": \"多用一点手腕的力量\",\
                                \"comment_en\": \"use more power on right wrist\"(A sample translation, maybe incorrect)\
                                \"suggestion_key\": \"right_wrist\",\
                                \"suggestion_type\": \"strength\",\
                                \"comment_index\": 2,\
                                \"stroke_index\": 12,\
                                \"type\": \"global\"\
                                }]\
                            </example3>\
                            <example4>Here is another example, input:  [{\
                                \"start\": \"00:00:49,659\",\
                                \"end\": \"00:00:54,479\",\
                                \"text\": \"整体动作的影拍低于台面\",\
                                \"index\": 3\
                                \"stroke_index\": 12,\
                                \"type\": \"realtime\"\
                            },\
                            {\
                                \"start\": \"00:00:54,479\",\
                                \"end\": \"00:01:00,060\",\
                                \"text\": \"应将影拍的动作向后，而不是向下\",\
                                \"index\": 4,\
                                \"stroke_index\": 13,\
                                \"type\": \"realtime\"\
                            }]\
                            ourput:[{\
                            \"start\": \"00:00:49,659\",\
                            \"end\": \"00:01:00,060\",\
                            \"text\": \"整体动作的引拍低于台面，应将引拍的动作向后，而不是向下\",\
                            \"comment_en\": \"the backswing should not be too low, below the plane,\
                                the backswing should swing the racket backward,\
                                not downward\"(A sample translation, maybe incorrect)\
                            \"suggestion_key\": \"backswing of racket\",\
                            \"suggestion_type\": \"racket\",\
                            \"comment_index\": 3,\
                            \"stroke_index\": 12,\
                            \"type\": \"realtime\"\
                            }]\
                            </example4>\
                            <notation>every record in the output need to follow the structure :\
                            [{\
                                    \"start\": \"00:01:16,019\",\
                                    \"end\": \"00:01:19,760\",\
                                    \"text\": \"然后重心稍微身体稍微向前轻一点\",\
                                    \"comment_en\": \"center of gravity should not lean backward, slightly lean forward\",\
                                    \"suggestion_key\": \"center_of_gravity\",\
                                    \"suggestion_type\": \"position\",\
                                    \"comment_index\": 18,\
                                    \"stroke_index\": 12,\
                                    \"type\": \"global\"\
                            },\
                            {\
                            \"start\": \"00:01:49,659\",\
                            \"end\": \"00:02:00,060\",\
                            \"text\": \"整体动作的引拍低于台面，应将引拍的动作向后，而不是向下\",\
                            \"comment_en\": \"the backswing should not be too low, below the plane,\
                                the backswing should swing the racket backward,\
                                not downward\"(A sample translation, maybe incorrect)\
                            \"suggestion_key\": \"backswing of racket\",\
                            \"suggestion_type\": \"racket\",\
                            \"comment_index\": 20,\
                            \"stroke_index\": 12,\
                            \"type\": \"realtime\"\
                            },<other elements similar to above>\
                            ]\
                            Other irrelevant variables should remain unchanged. Focus on processing \"comment_en\"\</notation>"
                },
                {
                    "role": "user",
                    "content": json_string
                }
            ],response_format=Comments,
        )
        text = response.choices[0].message.content
        print(text)
        try:
            if not os.path.exists(target_folder):
                os.makedirs(target_folder, exist_ok=True)
        except Exception as e:
            print(f"An error occurred: {e}")

        output_file = os.path.join(str(target_folder), str(os.path.basename(test_file).removesuffix(".json")) + ".json")
        print('output', output_file)

        if os.path.exists(output_file):
            os.remove(output_file)

        with open(output_file, mode='w', encoding='utf-8') as f:
            json.dump(json.loads(text)["comments"], f, ensure_ascii=False, indent=4)

import tqdm

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(
        api_key=api_key
    )
    root_path = '../../sampledata'
    text_summarize(client, root_path)
