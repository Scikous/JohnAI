from pydantic import BaseModel, Field, field_validator, AfterValidator
from interfaces.vllm_interface import JohnVLLM
from interfaces.base import BaseModelConfig
from typing import List, Literal, Annotated, Optional


#to be dialogue options based schema
class Dialogue(BaseModel):
    """
    Represents the state of an in-game dialogue box.
    """
    is_dialogue_active: bool = Field(
        ..., 
        description="Is a dialogue box currently visible on the screen?"
    )
    speaker_name: Optional[str] = Field(
        None, 
        description="The name of the character currently speaking. Null if no speaker is shown."
    )
    dialogue_text: Optional[str] = Field(
        None, 
        description="The full text content currently visible in the dialogue box."
    )
    has_more_text_indicator: bool = Field(
        False, 
        description="True if there is a visual cue (e.g., a blinking cursor, an arrow) indicating more text will appear."
    )
    player_choices: List[str]
    selected_choice: Optional[str] = Field(
        None,
        description="The currently highlighted player choice. Null if no choices are present."
    )
    menu_layout: Optional[str]



from PIL import Image
import json
def dummy_keyboard(action, action_repeat):
    return f"{action} {action_repeat} times"

def dummy_controller(cur_action_index, next_action_index, layout):
    index_diff = cur_action_index - next_action_index
    repeat_action_times = abs(index_diff)

    if repeat_action_times == 0:
        return dummy_keyboard("interact", 1)
    elif layout == "horizontal":
        action_direction = "move_left" if index_diff > 0 else "move_right"
        return dummy_keyboard(action_direction, repeat_action_times)
    elif layout == "vertical":
        action_direction = "move_up" if index_diff > 0 else "move_down"        
        return dummy_keyboard(action_direction, repeat_action_times)

ass_p = "{\n"
def analyze_game_info(prompt, images, schema):
    
    guided_json_config = {
        "max_tokens": 1028,
        "temperature": 0.2,
        # "enable_thinking": False
        "skip_special_tokens": False,
        "guided_decoding": {
            "json": schema  # The key 'json' specifies the type
        }
    }

    resp = llm.dialogue_generator(prompt=prompt, assistant_prompt=ass_p, images=images, generation_config=guided_json_config, add_generation_prompt=False, continue_final_message=True)
    print('@'*100, '\n',resp)#  '\n-----\n', next_action)
    game_info = json.loads(resp)
    return game_info

def action_in_options(prompt, images, options):
    act_regex = "|".join(options)
    #action specific generation configuration
    guided_json_config = {
        "max_tokens": 1028,
        "temperature": 0.2,
        # "enable_thinking": False
        "skip_special_tokens": False,
        "guided_decoding": {
            "regex": act_regex  # The key 'json' specifies the type
        }
    }

    return llm.dialogue_generator(prompt=prompt, assistant_prompt=ass_p, images=images, generation_config=guided_json_config, add_generation_prompt=False, continue_final_message=True)
    
def dialogue_handler():
    json_schema = Dialogue.model_json_schema()
    print(json_schema)

    #does a move just to make sure there if there are player choices available, at least one will be highlighted by default
    def dummy_move():
        dummy_keyboard("move_down", 1)
        dummy_keyboard("move_up", 1)
        dummy_keyboard("move_right", 1)
        dummy_keyboard("move_left", 1)

    dummy_move()
    #analyze game screenshot
    images = [Image.open("debug_frames/dialogue-20.png")]#[Image.open("debug_frames/Memes-02-08-7_1.png")]
    anal_prompt = "You are a Gaming AI who is currently playing a video game. Analyze the current state of the game. Provide a JSON of your observations (only the ones relevant to playing the game)."
    game_info = analyze_game_info(anal_prompt, images, json_schema)
    print(game_info)
    action_options = game_info["player_choices"]
    
    if action_options:
        # cur_selection_index = menu_options.index(selected_option)
        cur_selection_index = action_options.index(game_info["selected_choice"])
        act_prompt = f"You are a Gaming AI who is currently playing a video game. Analyze the current state of the game.\nAvailable options:{action_options}\n\n Choose a single action based on the available options."
        resp_act = action_in_options(act_prompt, images, action_options)
        print("---"*100, '\n', resp_act)
        next_action = resp_act
        next_action_index = action_options.index(next_action)
        menu_layout = game_info["menu_layout"]
        print(dummy_controller(cur_selection_index, next_action_index, layout='vertical'))
    else:
        cur_selection_index, next_action_index = 0, 0
        menu_layout = None
        print(dummy_controller(cur_selection_index, next_action_index, menu_layout))

model_init_kwargs = {"gpu_memory_utilization": 0.93, "max_model_len": 8000, "trust_remote_code": True,
    }
model_config = BaseModelConfig(model_path_or_id="Qwen/Qwen3-VL-8B-Instruct-FP8", is_vision_model=True, uses_special_chat_template=False, model_init_kwargs=model_init_kwargs)
llm = JohnVLLM(model_config).load_model(model_config)
dialogue_handler()
