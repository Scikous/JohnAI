from pydantic import BaseModel, Field, field_validator, AfterValidator
from interfaces.vllm_interface import JohnVLLM
from interfaces.base import BaseModelConfig
from typing import List, Literal, Annotated, Optional


#to be menu options based schema
class Info(BaseModel):
    menu_layout: str
    menu_options: List[str]
    selected_option: str


#use for async streaming only
# from vllm.sampling_params import RequestOutputKind
# "output_kind": RequestOutputKind.DELTA,


from PIL import Image
import json
# images = [Image.open("Sagex/gaming/nier.webp")]#[Image.open("Sagex/gaming/nier.webp"), Image.open("Sagex/gaming/nier.webp")]
# # resp = l.dialogue_generator("Generate a new person:", generation_config=guided_json_config)

#Paper Lily -- only a testing temp one
control_schema ={
    "move_up": "moves the player forward or moves selection up",
    "move_down": "moves the player down/backwards or moves selection down",
    "move_left": "moves the player left or moves selection left",
    "move_right": "moves the player right or moves selection right",
    "interact": "interacts with the environment",
    #add combat related values.

}



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
    


def main_menu_handler():

    json_schema = Info.model_json_schema()
    print(json_schema)

    #analyze game screenshot
    images = [Image.open("debug_frames/img2.png")]#[Image.open("debug_frames/Memes-02-08-7_1.png")]
    anal_prompt = "You are a Gaming AI who is currently playing a video game. Analyze the current state of the game. Provide a JSON of your observations (only the ones relevant to playing the game, observations should also include spatial layout (vertical/horizontal/grid) if applicable)."
    game_info = analyze_game_info(anal_prompt, images, json_schema)

    ## paperlily main menu specific
    #extract relevant game information
    menu_options = game_info["menu_options"]
    menu_layout = game_info["menu_layout"]

    # select action based on extracted game info
    act_prompt = f"You are a Gaming AI who is currently playing a video game. Analyze the current state of the game.\nAvailable options:{menu_options}\n\n Choose a single action based on the available options."
    #image maybe maybe not needed, unsure
    # images = [Image.open("debug_frames/img2.png")]#[Image.open("debug_frames/Memes-02-08-7_1.png")]
    resp_act = action_in_options(act_prompt, images, menu_options)
    print("@"*100, '\n',resp_act)

    #dummy controller takes action in MAIN MENU
    selected_option = game_info["selected_option"]
    next_action = resp_act
    next_action_index = menu_options.index(next_action)
    cur_selection_index = menu_options.index(selected_option)
    print(dummy_controller(cur_selection_index, next_action_index, menu_layout))

model_init_kwargs = {"gpu_memory_utilization": 0.93, "max_model_len": 8000, "trust_remote_code": True,
    }
model_config = BaseModelConfig(model_path_or_id="Qwen/Qwen3-VL-8B-Instruct-FP8", is_vision_model=True, uses_special_chat_template=False, model_init_kwargs=model_init_kwargs)
llm = JohnVLLM(model_config).load_model(model_config)

main_menu_handler()