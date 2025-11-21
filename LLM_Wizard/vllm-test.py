# from vllm import LLM, SamplingParams
# from vllm.sampling_params import GuidedDecodingParams
# from pydantic import BaseModel, Field, field_validator, AfterValidator
# from interfaces.vllm_interface import VtuberVLLM
# from interfaces.base import BaseModelConfig
# from typing import List, Literal, Annotated, Optional

# # def action_in_options(v, values):
# #     options = values.data.get('menu_options')
# #     raise ValueError(f"Action '{v}' is not one of the valid options: {options}")
# #     if (options and v not in options):
# #         pass
# #     return v

# #to be menu options based schema
# class Info(BaseModel):
#     menu_layout: str
#     menu_options: List[str]
#     selected_option: str

# #to be dialogue options based schema
# class Dialogue(BaseModel):
#     """
#     Represents the state of an in-game dialogue box.
#     """
#     is_dialogue_active: bool = Field(
#         ..., 
#         description="Is a dialogue box currently visible on the screen?"
#     )
#     speaker_name: Optional[str] = Field(
#         None, 
#         description="The name of the character currently speaking. Null if no speaker is shown."
#     )
#     dialogue_text: Optional[str] = Field(
#         None, 
#         description="The full text content currently visible in the dialogue box."
#     )
#     has_more_text_indicator: bool = Field(
#         False, 
#         description="True if there is a visual cue (e.g., a blinking cursor, an arrow) indicating more text will appear."
#     )
#     player_choices: List[str]
#     selected_choice: Optional[str] = Field(
#         None,
#         description="The currently highlighted player choice. Null if no choices are present."
#     )
#     menu_layout: Optional[str]

# class Movement(BaseModel):
#     pass


# ##create combat options based schema
# class Combat(BaseModel):
#     menu_layout: str
#     menu_options: List[str]
#     selected_option: str
# # #change based on personal needs
# # model_init_kwargs = {
# #     "gpu_memory_utilization": 0.95,
# #     "max_model_len": 49152
# #     }
# # config = BaseModelConfig(model_init_kwargs=model_init_kwargs)
# # l = VtuberVLLM(config).load_model(config)

# def image_resizer():
#     pass # TODO -- images MUST BE 1024x1024 for functional results.



# #zai-org/GLM-4.1V-9B-Thinking
# model_init_kwargs = {"gpu_memory_utilization": 0.93, "max_model_len": 8000, "trust_remote_code": True,
#     }
# model_config = BaseModelConfig(model_path_or_id="Qwen/Qwen3-VL-8B-Instruct-FP8", is_vision_model=True, uses_special_chat_template=False, model_init_kwargs=model_init_kwargs)
# llm = VtuberVLLM(model_config).load_model(model_config)

# #use for async streaming only
# # from vllm.sampling_params import RequestOutputKind
# # "output_kind": RequestOutputKind.DELTA,


# from PIL import Image
# import json
# # images = [Image.open("LLM_Wizard/gaming/nier.webp")]#[Image.open("LLM_Wizard/gaming/nier.webp"), Image.open("LLM_Wizard/gaming/nier.webp")]
# # # resp = l.dialogue_generator("Generate a new person:", generation_config=guided_json_config)

# #Paper Lily -- only a testing temp one
# control_schema ={
#     "move_up": "moves the player forward or moves selection up",
#     "move_down": "moves the player down/backwards or moves selection down",
#     "move_left": "moves the player left or moves selection left",
#     "move_right": "moves the player right or moves selection right",
#     "interact": "interacts with the environment",
#     #add combat related values.

# }

# # #Adventure, RPG,etc. -- other types as well
# # control_schema ={

# # }


# def dummy_keyboard(action, action_repeat):
#     return f"{action} {action_repeat} times"

# def dummy_controller(cur_action_index, next_action_index, layout):
#     index_diff = cur_action_index - next_action_index
#     repeat_action_times = abs(index_diff)

#     if repeat_action_times == 0:
#         return dummy_keyboard("interact", 1)
#     elif layout == "horizontal":
#         action_direction = "move_left" if index_diff > 0 else "move_right"
#         return dummy_keyboard(action_direction, repeat_action_times)
#     elif layout == "vertical":
#         action_direction = "move_up" if index_diff > 0 else "move_down"        
#         return dummy_keyboard(action_direction, repeat_action_times)

# # def dummy_controller_super(act_options, current_option=None, selected_option=None, layout=None):
    
# #     if current_option is None:

# #     cur_action_index = act_options.index(current_option)
# #     next_action_index = act_options.index(selected_option)
# #     repeat_action_times = abs(cur_action_index-next_action_index)

# #     if repeat_action_times == 0:
# #         return dummy_keyboard("interact", 1)
# #     elif layout == "horizontal":
# #         return dummy_keyboard("move_left", repeat_action_times)
# #     elif layout == "vertical":
# #         return dummy_keyboard("move_up", repeat_action_times)

# # #analysis
# ass_p = "{\n"
# def analyze_game_info(prompt, images, schema):
    
#     guided_json_config = {
#         "max_tokens": 1028,
#         "temperature": 0.2,
#         # "enable_thinking": False
#         "skip_special_tokens": False,
#         # "guided_decoding": {
#         #     "json": schema  # The key 'json' specifies the type
#         # }
#     }

#     resp = llm.dialogue_generator(prompt=prompt, assistant_prompt=ass_p, images=images, generation_config=guided_json_config, add_generation_prompt=False, continue_final_message=True)
#     print('@'*100, '\n',resp)#  '\n-----\n', next_action)
#     game_info = json.loads(resp)
#     return game_info

# def action_in_options(prompt, images, options):
#     act_regex = "|".join(options)
#     #action specific generation configuration
#     guided_json_config = {
#         "max_tokens": 1028,
#         "temperature": 0.2,
#         # "enable_thinking": False
#         "skip_special_tokens": False,
#         "guided_decoding": {
#             "regex": act_regex  # The key 'json' specifies the type
#         }
#     }

#     return llm.dialogue_generator(prompt=prompt, assistant_prompt=ass_p, images=images, generation_config=guided_json_config, add_generation_prompt=False, continue_final_message=True)
    

# #intermediate prompt assuming next action/desire is to be asked here (i.e. not the first prompt)



# def main_menu_handler():

#     json_schema = Info.model_json_schema()
#     print(json_schema)

#     #analyze game screenshot
#     images = [Image.open("debug_frames/img2.png")]#[Image.open("debug_frames/Memes-02-08-7_1.png")]
#     anal_prompt = "You are a Gaming AI who is currently playing a video game. Analyze the current state of the game. Provide a JSON of your observations (only the ones relevant to playing the game, observations should also include spatial layout (vertical/horizontal/grid) if applicable)."
#     game_info = analyze_game_info(anal_prompt, images, json_schema)

#     ## paperlily main menu specific
#     #extract relevant game information
#     menu_options = game_info["menu_options"]
#     menu_layout = game_info["menu_layout"]

#     # select action based on extracted game info
#     act_prompt = f"You are a Gaming AI who is currently playing a video game. Analyze the current state of the game.\nAvailable options:{menu_options}\n\n Choose a single action based on the available options."
#     #image maybe maybe not needed, unsure
#     # images = [Image.open("debug_frames/img2.png")]#[Image.open("debug_frames/Memes-02-08-7_1.png")]
#     resp_act = action_in_options(act_prompt, images, menu_options)
#     print("@"*100, '\n',resp_act)

#     #dummy controller takes action in MAIN MENU
#     selected_option = game_info["selected_option"]
#     next_action = resp_act
#     next_action_index = menu_options.index(next_action)
#     cur_selection_index = menu_options.index(selected_option)
#     print(dummy_controller(cur_selection_index, next_action_index, menu_layout))


# def dialogue_handler():
#     json_schema = Dialogue.model_json_schema()
#     print(json_schema)

#     #does a move just to make sure there if there are player choices available, at least one will be highlighted by default
#     def dummy_move():
#         dummy_keyboard("move_down", 1)
#         dummy_keyboard("move_up", 1)
#         dummy_keyboard("move_right", 1)
#         dummy_keyboard("move_left", 1)

#     dummy_move()
#     #analyze game screenshot
#     images = [Image.open("debug_frames/dialogue-20.png")]#[Image.open("debug_frames/Memes-02-08-7_1.png")]
#     anal_prompt = "You are a Gaming AI who is currently playing a video game. Analyze the current state of the game. Provide a JSON of your observations (only the ones relevant to playing the game)."
#     game_info = analyze_game_info(anal_prompt, images, json_schema)
#     print(game_info)
#     action_options = game_info["player_choices"]
    
#     if action_options:
#         # cur_selection_index = menu_options.index(selected_option)
#         cur_selection_index = action_options.index(game_info["selected_choice"])
#         act_prompt = f"You are a Gaming AI who is currently playing a video game. Analyze the current state of the game.\nAvailable options:{action_options}\n\n Choose a single action based on the available options."
#         resp_act = action_in_options(act_prompt, images, action_options)
#         print("---"*100, '\n', resp_act)
#         next_action = resp_act
#         next_action_index = action_options.index(next_action)
#         menu_layout = game_info["menu_layout"]
#         print(dummy_controller(cur_selection_index, next_action_index, layout='vertical'))
#     else:
#         cur_selection_index, next_action_index = 0, 0
#         menu_layout = None
#         print(dummy_controller(cur_selection_index, next_action_index, menu_layout))


# def movement_handler():
#     json_schema = Movement.model_json_schema()
#     print(json_schema)

#     #analyze game screenshot
#     images = [Image.open("debug_frames/stitched_panorama.png")]#[Image.open("debug_frames/Memes-02-08-7_1.png")]
#     anal_prompt = "You are a Gaming AI who is currently playing a video game. Analyze the current state of the game. Provide a JSON of your observations (only the ones relevant to playing the game). For objects and characters, include bounding boxes."
#     # anal_prompt = "You are a Gaming AI who is currently playing a video game. Find the player character(s) and estimate their displacement. Provide a JSON of the displacement."
#     game_info = analyze_game_info(anal_prompt, images, json_schema)
#     print(game_info)
#     # action_options = game_info["player_choices"]
    
#     # if action_options:
#     #     # cur_selection_index = menu_options.index(selected_option)
#     #     cur_selection_index = action_options.index(game_info["selected_choice"])
#     #     act_prompt = f"You are a Gaming AI who is currently playing a video game. Analyze the current state of the game.\nAvailable options:{action_options}\n\n Choose a single action based on the available options."
#     #     resp_act = action_in_options(act_prompt, images, action_options)
#     #     print("---"*100, '\n', resp_act)
#     #     next_action = resp_act
#     #     next_action_index = action_options.index(next_action)
#     #     menu_layout = game_info["menu_layout"]
#     #     print(dummy_controller(cur_selection_index, next_action_index, layout='vertical'))
#     # else:
#     #     cur_selection_index, next_action_index = 0, 0
#     #     menu_layout = None
#     #     print(dummy_controller(cur_selection_index, next_action_index, menu_layout))
        


# # main_menu_handler()
# # dialogue_handler()
# movement_handler()













# ## bbox acc checking
# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread("debug_frames/lacie-test.png")  # Replace with your image path

# # Define the bounding boxes from the JSON
# boxes = [
#     {"label": "player", "bbox": [476, 376, 523, 496], "color": (0, 255, 0)},
#     {"label": "bird", "bbox": [343, 616, 376, 665], "color": (0, 0, 255)},
#     {"label": "birdhouse", "bbox": [208, 442, 243, 616], "color": (255, 0, 0)},
#     # {"label": "tree", "bbox": [100, 273, 267, 513], "color": (0, 255, 255)},
#     {"label": "tree", "bbox": [0, 0, 267, 640], "color": (255, 255, 0)},
#     {"label": "stone_path", "bbox": [434, 256, 565, 636], "color": (128, 0, 255)},
#     {"label": "house", "bbox": [376, 0, 612, 276], "color": (255, 128, 0)},
#     # {"label": "brick_wall", "bbox": [690, 214, 885, 536], "color": (128, 128, 255)},
#     {"label": "fence", "bbox": [706, 650, 862, 773], "color": (0, 128, 255)},
#     {"label": "gate", "bbox": [455, 640, 536, 773], "color": (255, 0, 255)},
#     {"label": "lamp_post", "bbox": [112, 583, 137, 902], "color": (128, 255, 0)},
#     # {"label": "lamp_post", "bbox": [414, 486, 434, 536], "color": (128, 255, 0)},
#     {"label": "lamp_post", "bbox": [562, 552, 583, 636], "color": (128, 255, 0)},
#     # {"label": "lamp_post", "bbox": [894, 507, 918, 686], "color": (128, 255, 0)}
# ]

# # Draw each bounding box
# for box in boxes:
#     x, y, w, h = box["bbox"]
#     color = box["color"]
#     label = box["label"]
#     cv2.rectangle(image, (x, y), (w, h), color, 2)
#     cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# # Save or display the image
# cv2.imwrite("game_screenshot_with_boxes.png", image)
# cv2.imshow("Image with Bounding Boxes", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #############################################################################

# def take_action(key_press):
#     from gaming.controls import InputControllerThread
#     # input_controller = InputController()
#     input_controller = InputControllerThread()
#     input_controller.start()
#     key_press = ["left"]
#     input_controller.execute_action({"type": "key_press", "details": {"key": key_press, "hold_time": 1.0}})
#     input_controller.action_queue.join()
#     input_controller.stop()
# from gaming.controls import InputControllerThread
# input_controller = InputControllerThread()
# input_controller.start()
# def take_action(manager):
#     key_press = ["left"]
#     # while img:
#     #     img = _wait_for_new_frame(manager)

#     #     print("OOOPS")
#     input_controller.execute_action({"type": "key_press", "details": {"key": key_press, "hold_time": 1.0}})

# ### improved capture action system -- reuseable?
# import time
# import os
# from typing import Optional
# import threading
# import logging

# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - [%(filename)s] %(message)s")

# # logging.basicConfig(level=logging.INFO)
# # We assume the new CaptureManager and its worker are in a file named 'game_capture.py'
# # Make sure that file is in the same directory or your Python path.
# from gaming.game_capture import CaptureManager, Image

# # --- Helper Functions ---

# def _wait_for_new_frame(manager: CaptureManager, timeout_sec: float = 5.0) -> Optional[Image.Image]:
#     """
#     Waits until a new frame is available from the capture manager.

#     This is a robust replacement for time.sleep() and manual queue checking.
#     It repeatedly asks for the latest frame until it receives one or times out.
#     """
    
#     return manager.get_latest_frame()
#     # start_time = time.time()
#     # while time.time() - start_time < timeout_sec:
#     #     frame = manager.get_latest_frame()
#     #     if frame:
#     #         print("Successfully received a new frame.")
#     #     # Wait a short moment before polling again to avoid a busy-loop
#     #     time.sleep(0.1)
#     # print(f"Warning: Timed out after {timeout_sec} seconds waiting for a new frame.")
#     # return None

# # --- Main Logic ---

# def capture_action():
#     """
#     Captures the screen before and after an action using the CaptureManager.
#     """
#     # 1. Initialize the CaptureManager
#     # The manager encapsulates all the complex setup of queues, events, and the worker process.
#     # We set a low FPS because we only need snapshots, not a smooth video.
#     capture_manager = CaptureManager(
#         fps=60,  # Corresponds to your original interval_sec=0.2
#         target_size=(1000, 1000),
#         source_type=1
#     )

#     # 2. Use a try...finally block to ensure the worker is always stopped
#     # action_thread = None
#     try:
#         capture_manager.start()
        
#         print("Capture worker started. Please approve the screen sharing dialog within 5 seconds...")
#         time.sleep(5.0)

#         # 2. Get the "before" image
#         print("Attempting to get the 'before' image...")
#         # before_img = _wait_for_new_frame(capture_manager)
#         # # while not before_img:
#         # #     before_img = _wait_for_new_frame(capture_manager)
        
#         # if not before_img:
#         #     print("Could not capture 'before' image. Aborting.")
#         #     return
        
#         # before_img.save('debug_frames/character_dataset/capture_before_action.png')

#         print(f"Successfully saved 'before' image to '")

#         # 3. Perform the action and get the thread
#         print("Attempting to perform the action...")
#         # action_thread = take_action(None)
#         print("Action thread started.")

#         # 4. Capture frames while the action is happening
#         frame_count = 0
#         templist= []
#         while input_controller.is_alive(): # is_alive() checks if the thread is still running.
#             if frame_count == 0:
#                 before_img = _wait_for_new_frame(capture_manager)
#                 before_img.save('debug_frames/character_dataset/capture_before_action.png')

#                 take_action(capture_manager)
#                 frame_count = 1
#             # print(f"Action in progress, capturing frame {frame_count}...")
#             during_img = _wait_for_new_frame(capture_manager)
#             if during_img:
#                 templist.append(during_img)

#                 # during_img.save(os.path.join('debug_frames/character_dataset', f'capture_during_action_{frame_count:03d}.png'))
#                 # frame_count += 1
#             else:
#                 pass
#                 # print("Could not get a frame during action, continuing...")
#             # print(f"Action in progress, {input_controller.action_queue.unfinished_tasks}...")
#             if input_controller.action_queue.unfinished_tasks == 0:
#                 break
        
#         print("Action thread has finished.", len(templist))
#         for img in templist:
#             frame_count += 1
#             img.save(os.path.join('debug_frames/character_dataset', f'during_action_{frame_count:03d}.png'))
#         # Wait for the action queue to be fully processed
#         input_controller.action_queue.join()

#         # 5. Get the "after" image
#         print("Attempting to get the 'after' image...")
#         # Add a small delay to ensure the visual state has updated
#         time.sleep(0.5)
#         after_img = _wait_for_new_frame(capture_manager)

#         if not after_img:
#             print("Could not capture 'after' image. Aborting.")
#             return
            
#         after_img.save('debug_frames/character_dataset/capture_after_action.png')
#         # after_img.save(os.path.join(before_after_dir, 'capture_after_action.png'))
#         print(f"\nSuccessfully saved 'after' image to'")

#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#     finally:
#         # Ensure the input controller and capture worker are stopped
#         if input_controller:
#             print("Closing input controller...")
#             input_controller.stop()
        
#         print("Shutting down capture worker...")
#         capture_manager.stop()
#         print("Shutdown complete.")


# capture_action()



# import time
# import os
# from typing import Optional, Dict, Any
# import logging
# from queue import Empty

# # Assuming your other files are in a 'gaming' directory
# from gaming.game_capture import CaptureManager, Image
# from gaming.controls import InputControllerThread

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(filename)s] %(message)s")

# # --- Utility Functions ---

# def _deserialize_frame(frame_data: Dict[str, Any]) -> Image.Image:
#     """Convenience function to turn a data packet from the queue back into a PIL Image."""
#     return Image.frombytes(
#         frame_data['mode'],
#         frame_data['size'],
#         frame_data['image_bytes']
#     )

# def _save_image_sequence(path: str, file_prefix: str, images: list[Image.Image]):
#     """Saves a list of images to a directory with a numbered sequence."""
#     os.makedirs(path, exist_ok=True)
#     for i, img in enumerate(images):
#         img.save(os.path.join(path, f'{file_prefix}_{i:04d}.png'))
#     logging.info(f"Saved {len(images)} frames to '{path}' with prefix '{file_prefix}'.")


# # --- Main Logic ---

# def capture_action():
#     """
#     Captures the screen before, during, and after an action with high fidelity.
#     """
#     # --- Configuration ---
#     ACTION_DURATION_SECONDS = 1.0
#     FPS = 60
#     OUTPUT_DIR = 'debug_frames/character_dataset'

#     # 1. Initialize Managers
#     # We set maxsize to FPS * duration + a small buffer to hold all expected frames.
#     capture_manager = CaptureManager(fps=FPS, target_size=(1000, 1000), source_type=1)
#     input_controller = InputControllerThread()
    
#     try:
#         # Start child processes/threads
#         input_controller.start()
#         capture_manager.start()
        
#         logging.info("Workers started. Please approve the screen sharing dialog within 5 seconds...")
#         time.sleep(5.0) # Wait for user approval

#         # 2. "Priming" Phase: Get the 'before' frame.
#         # This ensures we get a truly fresh frame right before the action.
#         logging.info("Draining stale frames and waiting for a fresh 'before' frame...")
#         # Drain the queue of any frames that accumulated during startup
#         while not capture_manager._image_data_queue.empty():
#             try:
#                 capture_manager._image_data_queue.get_nowait()
#             except Empty:
#                 break
#         # Now, block and wait for the very next fresh frame
#         before_frame_data = capture_manager._image_data_queue.get(timeout=2.0)
#         logging.info("Successfully captured 'before' frame.")

#         # 3. Action & High-Fidelity Capture Phase
#         key_press_action = {
#             "type": "key_press",
#             "details": {"key": ["left"], "hold_time": ACTION_DURATION_SECONDS}
#         }
        
#         during_frames_data = []
        
#         logging.info(f"Dispatching action and starting high-speed capture for {ACTION_DURATION_SECONDS}s.")
        
#         # --- Synchronization Point ---
#         action_start_time = time.perf_counter()
#         # input_controller.execute_action(key_press_action)
        
#         # --- High-Speed Capture Loop ---
#         while (time.perf_counter() - action_start_time) < ACTION_DURATION_SECONDS:
#             try:
#                 # Use non-blocking get to drain the queue as fast as possible
#                 frame_data = capture_manager._image_data_queue.get_nowait()
#                 during_frames_data.append(frame_data)
#             except Empty:
#                 # This is expected and good! It means our loop is faster than the producer.
#                 # We do not sleep here, to minimize latency and be ready for the next frame.
#                 pass
        
#         logging.info(f"High-speed capture finished. Captured {len(during_frames_data)} frames.")

#         # 4. Finalizing Phase: Get the 'after' frame
#         # Block and wait for the first frame generated after the action window.
#         logging.info("Waiting for a fresh 'after' frame...")
#         after_frame_data = capture_manager._image_data_queue.get(timeout=2.0)
#         logging.info("Successfully captured 'after' frame.")

#         # 5. Post-Processing & Shutdown Phase
#         # Now that the time-critical work is done, we can do slow I/O operations.
#         logging.info("Processing and saving captured frames...")
#         print(len(during_frames_data))
#         # First, deserialize all the raw frame data into PIL Images
#         # before_img = _deserialize_frame(before_frame_data)
#         # during_imgs = [_deserialize_frame(data) for data in during_frames_data]
#         # after_img = _deserialize_frame(after_frame_data)

#         # # Now, save them to disk
#         # os.makedirs(OUTPUT_DIR, exist_ok=True)
#         # before_img.save(os.path.join(OUTPUT_DIR, 'capture_before_action.png'))
#         # after_img.save(os.path.join(OUTPUT_DIR, 'capture_after_action.png'))
#         # _save_image_sequence(OUTPUT_DIR, 'during_action', during_imgs)

#         # logging.info("All frames have been saved.")

#     except Empty:
#         logging.error("Timed out waiting for a frame from the capture worker. Is it running correctly?")
#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}", exc_info=True)
#     finally:
#         # Always ensure workers are stopped
#         if input_controller.is_alive():
#             logging.info("Stopping input controller...")
#             input_controller.stop()
        
#         logging.info("Stopping capture manager...")
#         capture_manager.stop()
#         logging.info("Shutdown complete.")


# if __name__ == "__main__":
#     capture_action()




# File: main_script.py

# File: main_script.py

# import time
# import os
# import logging
# import numpy as np
# from PIL import Image
# import cv2 # We need this for the color conversion

# # --- Use the FINAL multiprocessing-based OBS capture module ---
# from gaming.game_capture import CaptureManager
# from gaming.controls import InputControllerThread

# # ... (logging setup and _save_image_sequence function are the same) ...
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(filename)s] %(message)s")

# def _save_image_sequence(path: str, file_prefix: str, images: list[Image.Image]):
#     os.makedirs(path, exist_ok=True)
#     for i, img in enumerate(images):
#         img.save(os.path.join(path, f'{file_prefix}_{i:04d}.png'))
#     logging.info(f"Saved {len(images)} frames to '{path}' with prefix '{file_prefix}'.")

# def capture_action():
#     # ... (Configuration is the same) ...
#     ACTION_DURATION_SECONDS = 1.0
#     OUTPUT_DIR = 'debug_frames/character_dataset'
#     VIRTUAL_CAMERA_INDEX = 0

#     capture_manager = CaptureManager(device_index=VIRTUAL_CAMERA_INDEX)
#     input_controller = InputControllerThread()
    
#     try:
#         input_controller.start()
#         capture_manager.start()
        
#         logging.info("Workers started. Ensure OBS Virtual Camera is running.")
#         time.sleep(2.0)

#         logging.info("Draining stale frames and waiting for a fresh 'before' frame...")
#         capture_manager.drain_queue()
#         before_frame = capture_manager.get_frame() # Returns a NumPy copy
#         logging.info("Successfully captured 'before' frame.")

#         key_press_action = {
#             "type": "key_press",
#             "details": {"key": ["left"], "hold_time": ACTION_DURATION_SECONDS}
#         }
        
#         during_frames = []
#         logging.info(f"Dispatching action and starting high-speed capture for {ACTION_DURATION_SECONDS}s.")
        
#         action_start_time = time.perf_counter()
#         input_controller.execute_action(key_press_action)
        
#         while (time.perf_counter() - action_start_time) < ACTION_DURATION_SECONDS:
#             # get_frame waits for a notification of a new, unique frame
#             frame = capture_manager.get_frame()
#             during_frames.append(frame)
        
#         logging.info(f"High-speed capture finished. Captured {len(during_frames)} frames.")

#         logging.info("Waiting for a fresh 'after' frame...")
#         time.sleep(0.2)
#         capture_manager.drain_queue()
#         after_frame = capture_manager.get_frame()
#         logging.info("Successfully captured 'after' frame.")

#         logging.info("Processing and saving captured frames...")
        
#         before_img = Image.fromarray(cv2.cvtColor(before_frame, cv2.COLOR_BGR2RGB))
#         during_imgs = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in during_frames]
#         after_img = Image.fromarray(cv2.cvtColor(after_frame, cv2.COLOR_BGR2RGB))

#         # ... (Saving logic is the same) ...
#         os.makedirs(OUTPUT_DIR, exist_ok=True)
#         before_img.save(os.path.join(OUTPUT_DIR, 'capture_before_action.png'))
#         after_img.save(os.path.join(OUTPUT_DIR, 'capture_after_action.png'))
#         _save_image_sequence(OUTPUT_DIR, 'during_action', during_imgs)
#         logging.info("All frames have been saved.")

#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}", exc_info=True)
#     finally:
#         if 'input_controller' in locals() and input_controller.is_alive():
#             input_controller.stop()
#         if 'capture_manager' in locals():
#             capture_manager.stop()
#         logging.info("Shutdown complete.")

# if __name__ == "__main__":
#     capture_action()


# File: main_script.py

# import time
# import os
# import logging
# import cv2
# from PIL import Image
# from queue import Empty

# # We use the proven high-performance engine
# from gaming.game_capture import CaptureEngine 
# from gaming.controls import InputControllerThread

# # ... (logging setup and _save_image_sequence function are the same) ...
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(filename)s] %(message)s")

# def _save_image_sequence(path: str, file_prefix: str, images: list[Image.Image]):
#     os.makedirs(path, exist_ok=True)
#     for i, img in enumerate(images):
#         img.save(os.path.join(path, f'{file_prefix}_{i:04d}.png'))
#     logging.info(f"Saved {len(images)} frames to '{path}' with prefix '{file_prefix}'.")

# def capture_action():
#     ACTION_DURATION_SECONDS = 1.0
#     OUTPUT_DIR = 'debug_frames/character_dataset'
#     VIRTUAL_CAMERA_INDEX = 0

#     capture_engine = None
#     input_controller = InputControllerThread()
    
#     try:
#         input_controller.start()
#         capture_engine = CaptureEngine(VIRTUAL_CAMERA_INDEX).start()
        
#         logging.info("Workers started. Waiting for capture stream to stabilize...")
#         # A short, one-time sleep here is acceptable to let the producer
#         # thread initialize and start the stream.
#         time.sleep(4.0) 

#         # --- THE "LIVE SYNC" AND CAPTURE OF 'before_frame' ---
#         logging.info("Synchronizing with live frame stream...")
        
#         # 1. Rapidly drain any frames that have buffered since the start.
#         #    This loop runs very fast.
#         while True:
#             try:
#                 # Use get_nowait() to empty the queue without blocking.
#                 capture_engine.q.get_nowait()
#             except Empty:
#                 # The queue is now empty. The very next frame is "live".
#                 break
        
#         # 2. Block and wait for the single, definitive "live" before_frame.
#         before_frame = capture_engine.read()
#         logging.info("Live sync complete. Captured 'before' frame.")

#         # --- The rest of the logic proceeds as before ---
        
#         key_press_action = {
#             "type": "key_press",
#             "details": {"key": ["left"], "hold_time": ACTION_DURATION_SECONDS}
#         }
        
#         during_frames = []
        
#         logging.info(f"Dispatching action and starting high-speed capture for {ACTION_DURATION_SECONDS}s.")
        
#         action_start_time = time.perf_counter()
#         # input_controller.execute_action(key_press_action)
        
#         # This loop will now run at 55-60 FPS because the producer is already
#         # in its high-performance state, and the queue is not empty when this
#         # loop begins (the producer will have added a frame while we were dispatching the action).
#         while (time.perf_counter() - action_start_time) < ACTION_DURATION_SECONDS:
#             frame = capture_engine.read()
#             during_frames.append(frame)
        
#         logging.info(f"High-speed capture finished. Captured {len(during_frames)} frames.")

#         # For the 'after_frame', we do the same sync process to ensure it's live.
#         logging.info("Synchronizing for 'after' frame...")
#         while True:
#             try:
#                 capture_engine.q.get_nowait()
#             except Empty:
#                 break
#         after_frame = capture_engine.read()
#         logging.info("Captured 'after' frame.")

#         # --- Post-Processing and Shutdown ---
#         logging.info("Processing and saving captured frames...")
#         print(len(during_frames))
#         # before_img = Image.fromarray(cv2.cvtColor(before_frame, cv2.COLOR_BGR2RGB))
#         # during_imgs = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in during_frames]
#         # after_img = Image.fromarray(cv2.cvtColor(after_frame, cv2.COLOR_BGR2RGB))

#         # os.makedirs(OUTPUT_DIR, exist_ok=True)
#         # before_img.save(os.path.join(OUTPUT_DIR, 'capture_before_action.png'))
#         # after_img.save(os.path.join(OUTPUT_DIR, 'capture_after_action.png'))
#         # _save_image_sequence(OUTPUT_DIR, 'during_action', during_imgs)
#         # logging.info("All frames have been saved.")

#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}", exc_info=True)
#     finally:
#         if 'input_controller' in locals() and input_controller.is_alive():
#             input_controller.stop()
#         if 'capture_engine' in locals():
#             capture_engine.stop()
#         logging.info("Shutdown complete.")

# if __name__ == "__main__":
#     capture_action()




# import time
# import os
# import logging
# import numpy as np
# from PIL import Image
# import cv2

# # --- NEW: Import from our reliable OBS capture module ---
# from gaming.game_capture import CaptureManager
# # --- We still use the input controller ---
# from gaming.controls import InputControllerThread

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(filename)s] %(message)s")

# def _save_image_sequence(path: str, file_prefix: str, images: list[Image.Image]):
#     """Saves a list of images to a directory with a numbered sequence."""
#     os.makedirs(path, exist_ok=True)
#     for i, img in enumerate(images):
#         img.save(os.path.join(path, f'{file_prefix}_{i:04d}.png'))
#     logging.info(f"Saved {len(images)} frames to '{path}' with prefix '{file_prefix}'.")

# def capture_action():
#     """
#     Captures screen frames before, during, and after an action using the
#     stable OBS Virtual Camera as the source.
#     """
#     # --- Configuration ---
#     ACTION_DURATION_SECONDS = 1.0
#     OUTPUT_DIR = 'debug_frames/character_dataset'
#     VIRTUAL_CAMERA_INDEX = 0 # Update this if needed (check with `v4l2-ctl --list-devices`)

#     # 1. Initialize Managers
#     # All the complex setup is now handled by our new CaptureManager.
#     capture_manager = CaptureManager(device_index=VIRTUAL_CAMERA_INDEX)
#     input_controller = InputControllerThread()
    
#     try:
#         # Start background threads
#         input_controller.start()
#         capture_manager.start()
        
#         logging.info("Workers started. Ensure OBS Virtual Camera is running.")
#         # Give the capture thread time to populate the queue.
#         time.sleep(2.0)

#         # 2. "Priming" Phase: Get the 'before' frame.
#         logging.info("Draining stale frames and waiting for a fresh 'before' frame...")
#         capture_manager.drain_queue()
#         before_frame = capture_manager.get_frame() # This is a NumPy array
#         if before_frame is None:
#             raise RuntimeError("Could not capture 'before' frame. Timed out.")
#         logging.info("Successfully captured 'before' frame.")

#         # 3. Action & High-Fidelity Capture Phase
#         key_press_action = {
#             "type": "key_press",
#             "details": {"key": ["left"], "hold_time": ACTION_DURATION_SECONDS}
#         }
        
#         during_frames = []
        
#         logging.info(f"Dispatching action and starting high-speed capture for {ACTION_DURATION_SECONDS}s.")
        
#         action_start_time = time.perf_counter()
#         input_controller.execute_action(key_press_action)
        
#         while (time.perf_counter() - action_start_time) < ACTION_DURATION_SECONDS:
#             # get_frame blocks until a new, unique frame is ready.
#             # This guarantees we capture the true sequence of frames.
#             frame = capture_manager.get_frame(timeout=0.01)
#             if frame is not None:
#                 during_frames.append(frame)
        
#         logging.info(f"High-speed capture finished. Captured {len(during_frames)} frames.")

#         # 4. Finalizing Phase: Get the 'after' frame
#         logging.info("Waiting for a fresh 'after' frame...")
#         # Add a small delay to ensure the visual state has updated post-action
#         time.sleep(0.2)
#         capture_manager.drain_queue() # Clear any frames captured between the loop ending and now
#         after_frame = capture_manager.get_frame()
#         if after_frame is None:
#             raise RuntimeError("Could not capture 'after' frame. Timed out.")
#         logging.info("Successfully captured 'after' frame.")

#         # 5. Post-Processing & Shutdown Phase
#         logging.info("Processing and saving captured frames...")
        
#         # OpenCV frames are BGR, so we convert them to RGB for PIL/saving.
#         before_img = Image.fromarray(cv2.cvtColor(before_frame, cv2.COLOR_BGR2RGB))
#         during_imgs = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in during_frames]
#         after_img = Image.fromarray(cv2.cvtColor(after_frame, cv2.COLOR_BGR2RGB))

#         os.makedirs(OUTPUT_DIR, exist_ok=True)
#         before_img.save(os.path.join(OUTPUT_DIR, 'capture_before_action.png'))
#         after_img.save(os.path.join(OUTPUT_DIR, 'capture_after_action.png'))
#         _save_image_sequence(OUTPUT_DIR, 'during_action', during_imgs)

#         logging.info("All frames have been saved.")

#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}", exc_info=True)
#     finally:
#         # Always ensure workers are stopped gracefully
#         if 'input_controller' in locals() and input_controller.is_alive():
#             logging.info("Stopping input controller...")
#             input_controller.stop()
        
#         if 'capture_manager' in locals():
#             logging.info("Stopping capture manager...")
#             capture_manager.stop()
            
#         logging.info("Shutdown complete.")


# if __name__ == "__main__":
#     # Ensure OBS is running and the Virtual Camera is started before running this script.
#     capture_action()






import time
import os
import logging
import cv2
import numpy as np
from typing import Optional

# --- Imports ---
# 1. The Optimized Capture System (saved from previous step)
from gaming.game_capture import CaptureManager, SystemConfig

# 2. Your Input Controller
try:
    from gaming.controls import InputControllerThread
except ImportError:
    # Mocking for demonstration if the file isn't present locally
    import threading
    class InputControllerThread(threading.Thread):
        def execute_action(self, action): print(f"[MockInput] Executing: {action}")
        def stop(self): pass
        def run(self): pass

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)


# hacky temporary fix for running with sudo -- gives permission of captured images to user.
def save_as_user(path, image):
    """Saves an image and immediately restores ownership to the non-root user."""
    # 1. Save the file (currently owned by root)
    cv2.imwrite(path, image)
    
    # 2. Check if we are running via sudo
    sudo_uid = os.environ.get('SUDO_UID')
    sudo_gid = os.environ.get('SUDO_GID')
    
    if sudo_uid and sudo_gid:
        try:
            # 3. Change ownership back to the original user
            os.chown(path, int(sudo_uid), int(sudo_gid))
        except Exception as e:
            logging.warning(f"Could not change file ownership: {e}")


def capture_action_sequence():
    """
    Orchestrates the Before -> Action -> After capture flow.
    """
    # --- Configuration ---
    ACTION_DURATION = 1.0
    OUTPUT_DIR = 'debug_frames/character_dataset'
    
    # Configure the system for VLM (1000x1000)
    config = SystemConfig(
        device_index=0,
        src_width=2560,
        src_height=1440,
        target_fps=60,
        target_size=(1000, 1000), # VLM Standard
        enable_psutil=True,
        warmup_time=2.0
    )

    manager = CaptureManager(config)
    input_controller = InputControllerThread()
    
    try:
        # 1. Start Background Processes (Includes Warmup)
        logging.info("System: Initializing workers...")
        input_controller.start()
        manager.start_system() # This blocks for 2.0s for warmup
        
        time.sleep(2.0)
        # 2. Capture "Before" State
        logging.info("Phase: Capturing 'Before' frame...")
        # We capture a tiny slice of time to ensure we get a fresh frame
        before_frame_raw = manager.get_snapshot()
        if before_frame_raw is None:
            raise RuntimeError("Failed to capture 'Before' frame.")

        # 3. Capture "Action" State
        logging.info(f"Phase: Executing Action for {ACTION_DURATION}s...")
        
        # Start filling the RAM buffer
        manager.start_capture()
        
        # Trigger the physical action
        start_t = time.perf_counter()
        input_controller.execute_action({
            "type": "key_press",
            "details": {"key": ["left"], "hold_time": ACTION_DURATION}
        })
        
        # Wait strictly for the duration
        # We calculate sleep to ensure exact timing, accounting for execution overhead
        elapsed = time.perf_counter() - start_t
        remaining = ACTION_DURATION - elapsed
        if remaining > 0:
            time.sleep(remaining)
            
        # Stop filling buffer
        during_frames_raw = manager.stop_capture()
        logging.info(f"Action captured: {len(during_frames_raw)} raw frames.")

        # 4. Capture "After" State
        # Wait a moment for physics/animations to settle
        time.sleep(0.2)
        
        logging.info("Phase: Capturing 'After' frame...")
        after_frame_raw = manager.get_snapshot()
        if after_frame_raw is None:
            raise RuntimeError("Failed to capture 'After' frame.")

        # 5. Post-Processing (The Heavy Lifting)
        logging.info("Phase: Post-Processing (Resizing & Letterboxing)...")
        
        # Combine everything into one list to maximize thread pool efficiency
        # Structure: [Before] + [During...] + [After]
        all_raw_frames = [before_frame_raw] + during_frames_raw + [after_frame_raw]
        
        t0 = time.perf_counter()
        # This runs the 1000x1000 letterbox logic on all cores
        all_processed = manager.post_process_frames(all_raw_frames)
        logging.info(f"Processed {len(all_processed)} frames in {time.perf_counter() - t0:.3f}s")

        # Separate them back out
        before_final = all_processed[0]
        during_final = all_processed[1:-1]
        after_final = all_processed[-1]

        # 6. Save to Disk
        # Note: cv2.imwrite expects BGR, which is what we have. No conversion needed.
        logging.info(f"Saving to {OUTPUT_DIR}...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        sudo_uid = os.environ.get('SUDO_UID')
        sudo_gid = os.environ.get('SUDO_GID')
        if sudo_uid and sudo_gid:
            os.chown(OUTPUT_DIR, int(sudo_uid), int(sudo_gid))

        # Use the helper function instead of cv2.imwrite directly -- TODO: drop when moving on.
        save_as_user(os.path.join(OUTPUT_DIR, 'capture_before_action.png'), before_final)
        save_as_user(os.path.join(OUTPUT_DIR, 'capture_after_action.png'), after_final)
        
        for i, frame in enumerate(during_final):
            fname = f"during_action_{i:04d}.png"
            save_as_user(os.path.join(OUTPUT_DIR, fname), frame)
        logging.info("Sequence complete.")

    except Exception as e:
        logging.error(f"Critical Error: {e}", exc_info=True)
    finally:
        # Clean shutdown
        if input_controller.is_alive():
            input_controller.stop()
        manager.stop_system()
        logging.info("System shutdown.")

if __name__ == "__main__":
    capture_action_sequence()



# import time
# time.sleep(5)
# take_action()
# ##########################################################################






# import pygame
# import math
# import random

# # --- 1. Core Component: A Vector2D Class ---
# # We need a simple class to handle 2D vectors for position, velocity, and forces.
# class Vector2D:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def __add__(self, other):
#         return Vector2D(self.x + other.x, self.y + other.y)

#     def __sub__(self, other):
#         return Vector2D(self.x - other.x, self.y - other.y)

#     def __mul__(self, scalar):
#         return Vector2D(self.x * scalar, self.y * scalar)

#     def magnitude(self):
#         return math.sqrt(self.x**2 + self.y**2)

#     def normalize(self):
#         mag = self.magnitude()
#         if mag > 0:
#             return Vector2D(self.x / mag, self.y / mag)
#         return Vector2D(0, 0)

#     def set_magnitude(self, new_mag):
#         return self.normalize() * new_mag

#     def limit(self, max_mag):
#         if self.magnitude() > max_mag:
#             return self.set_magnitude(max_mag)
#         return self

# # --- 2. The Agent (Our "Pilot") ---
# class Agent:
#     def __init__(self, x, y):
#         self.position = Vector2D(x, y)
#         self.velocity = Vector2D(0, 0)
#         self.acceleration = Vector2D(0, 0)
#         self.radius = 10
#         self.max_speed = 1  # Maximum speed in pixels per frame
#         self.max_force = 0.15 # Maximum steering force to apply

#     def apply_force(self, force):
#         # Newton's second law (F=ma), but we assume mass=1, so F=a
#         self.acceleration += force

#     def update(self):
#         # Update velocity based on acceleration
#         self.velocity += self.acceleration
#         # Limit velocity to max_speed
#         self.velocity = self.velocity.limit(self.max_speed)
#         # Update position based on velocity
#         self.position += self.velocity
#         # Reset acceleration for the next frame
#         self.acceleration = self.acceleration * 0

#     def draw(self, screen):
#         # Draw the agent as a circle
#         pygame.draw.circle(screen, (255, 255, 255), (int(self.position.x), int(self.position.y)), self.radius)
#         # Draw a line indicating direction
#         end_pos = self.position + self.velocity.normalize() * 15
#         pygame.draw.line(screen, (255, 0, 0), (self.position.x, self.position.y), (end_pos.x, end_pos.y), 2)


#     # --- 3. The Steering Behaviors ---

#     def arrive(self, target_pos):
#         # This behavior steers the agent to a target and slows down as it approaches.
#         desired_velocity = target_pos - self.position
#         distance = desired_velocity.magnitude()

#         slowing_radius = 100 # The radius within which the agent starts to slow down

#         if distance < slowing_radius:
#             # If inside the slowing radius, map the distance to a speed
#             desired_speed = (distance / slowing_radius) * self.max_speed
#             desired_velocity = desired_velocity.set_magnitude(desired_speed)
#         else:
#             # Otherwise, move at max speed
#             desired_velocity = desired_velocity.set_magnitude(self.max_speed)

#         # The core of steering: Steering Force = Desired Velocity - Current Velocity
#         steering_force = desired_velocity - self.velocity
#         steering_force = steering_force.limit(self.max_force)
#         return steering_force

#     def obstacle_avoidance(self, obstacles):
#         # This behavior steers the agent to avoid a list of obstacles.
#         total_avoidance_force = Vector2D(0, 0)
#         avoidance_radius = 50 # How far ahead the agent "looks" for obstacles

#         for obstacle in obstacles:
#             dist_to_obstacle = (obstacle.position - self.position).magnitude()

#             # Only consider obstacles that are close
#             if 0 < dist_to_obstacle < avoidance_radius:
#                 # Fleeing force is stronger the closer the agent is to the obstacle
#                 flee_force = self.position - obstacle.position
#                 # Scale the force inversely to the distance
#                 flee_force = flee_force.set_magnitude(self.max_force * (1 - (dist_to_obstacle / avoidance_radius)))
#                 total_avoidance_force += flee_force

#         return total_avoidance_force.limit(self.max_force * 2) # Avoidance can be a stronger force


#     def combine_forces(self, target, obstacles):
#         # Weights determine the priority of each behavior.
#         # Here, avoidance is much more important than arriving.
#         arrive_weight = 0.5
#         avoidance_weight = 2.0

#         arrive_force = self.arrive(target) * arrive_weight
#         avoidance_force = self.obstacle_avoidance(obstacles) * avoidance_weight

#         # Apply the combined forces
#         self.apply_force(arrive_force)
#         self.apply_force(avoidance_force)


# class Obstacle:
#     def __init__(self, x, y, radius):
#         self.position = Vector2D(x, y)
#         self.radius = radius

#     def draw(self, screen):
#         pygame.draw.circle(screen, (100, 100, 255), (int(self.position.x), int(self.position.y)), self.radius)


# # --- 4. The Main Simulation ---
# def run_simulation():
#     pygame.init()
#     width, height = 1000, 1000
#     screen = pygame.display.set_mode((width, height))
#     pygame.display.set_caption("Steering Behavior Simulation")
#     clock = pygame.time.Clock()

#     # Create our agent and obstacles
#     agent = Agent(499, 436)#(width / 2, height / 2)
#     obstacles = [Obstacle(random.randint(100, 700), random.randint(100, 500), 30) for _ in range(7)]
#     obstacles.append(Obstacle(429, 540, 50))
#     target_pos = Vector2D(359,640)

#     running = True
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             # The VLM would update this target, but for now, we'll use the mouse
#             if event.type == pygame.MOUSEBUTTONDOWN:
#                 mx, my = pygame.mouse.get_pos()
#                 target_pos = Vector2D(mx, my)

#         # --- Core Logic ---
#         # 1. Calculate and combine forces
#         agent.combine_forces(target_pos, obstacles)

#         # 2. Update agent's position
#         agent.update()

#         # 3. Keep agent on screen (wrapping)
#         if agent.position.x > width: agent.position.x = 0
#         if agent.position.x < 0: agent.position.x = width
#         if agent.position.y > height: agent.position.y = 0
#         if agent.position.y < 0: agent.position.y = height


#         # --- Drawing ---
#         screen.fill((20, 20, 20)) # Dark background

#         # Draw the target
#         pygame.draw.circle(screen, (0, 255, 0), (int(target_pos.x), int(target_pos.y)), 15)
#         pygame.draw.circle(screen, (255,255,255), (int(target_pos.x), int(target_pos.y)), 15, 2)


#         # Draw obstacles and the agent
#         for obstacle in obstacles:
#             obstacle.draw(screen)
#         agent.draw(screen)

#         pygame.display.flip()
#         clock.tick(60) # Limit to 60 FPS

#     pygame.quit()

# if __name__ == '__main__':
#     run_simulation()






# #TODO: hold_time and probs press_threshold needa be modifieable/calculatable.
# class ActionTranslator:
#     """Translates a 2D intention vector into a discrete keyboard action."""
#     def __init__(self, press_threshold=0.3, hold_time=0.1):
#         """
#         Initializes the translator.
#         Args:
#             press_threshold (float): How strong the vector's component must be to trigger a key press.
#             hold_time (float): The duration in seconds for the key press action.
#         """
#         self.press_threshold = press_threshold
#         self.hold_time = hold_time

#     def translate(self, vector, directional_constraints):
#         """
#         Converts the vector into an action dictionary.
#         Args:
#             vector (Vector2D): The intention vector from the MovementController.
#             directional_constraints (str): e.g., 'eight_way', 'four_way', 'horizontal_only'.
        
#         Returns:
#             dict or None: An action dictionary like {"key_press": ["w", "a"], "hold_time": 0.1} or None if no action.
#         """
#         keys_to_press = []

#         # Horizontal Movement
#         if vector.x < -self.press_threshold:
#             keys_to_press.append("left")
#         elif vector.x > self.press_threshold:
#             keys_to_press.append("right")

#         # Vertical Movement
#         if directional_constraints in ['eight_way', 'four_way', 'vertical_only']:
#             if vector.y < -self.press_threshold:
#                 keys_to_press.append("up")
#             elif vector.y > self.press_threshold:
#                 keys_to_press.append("down")

#         # Handle 'four_way' constraint (no diagonals)
#         if directional_constraints == 'four_way' and len(keys_to_press) > 1:
#             # Prioritize the direction with the stronger intention
#             if abs(vector.x) > abs(vector.y):
#                 keys_to_press = [k for k in keys_to_press if k in ["left", "right"]]
#             else:
#                 keys_to_press = [k for k in keys_to_press if k in ["up", "down"]]
        
#         if not keys_to_press:
#             return None

#         return {"key_press": sorted(keys_to_press), "hold_time": self.hold_time}



# if __name__ == '__main__':
#     # --- 1. Define NEW Game Configuration ---
#     # Now includes the real-world pixels-per-second speed.
#     game_config = {
#         'movement_mode': 'constant_speed',
#         'directional_constraints': 'eight_way',
#         'max_speed_pps': 193.82, # The known horizontal speed
#         'max_tick_speed': 10.0 # How many pixels the agent wants to move per decision tick
#     }

#     # --- 2. Initialize Controller and Game State ---
#     print("--- SIMULATION: Calibrated Movement with Arrival Logic ---")
#     controller = MovementController(game_config)
    
#     character_position = Vector2D(499, 436)
#     target_position = Vector2D(359, 640)
#     obstacles = [{'position': Vector2D(250, 220), 'radius': 60, 'impenetrable': True}]

#     # --- 3. Run The Simulation Loop ---
#     time.sleep(5.0)
#     for i in range(500):
#         # MODIFIED: Pass the interaction_range to the decision maker
#         # For an NPC, this might be 30px, for an item, 5px.
#         action = controller.decide_action(
#             character_position, 
#             target_position, 
#             obstacles, 
#             interaction_range=50.0
#         )
        
#         # --- Mock Game Engine ---
#         print(f"TickAA {i+1:03d} | Pos: {character_position} | Action: {action}")
        
#         if action:
#             # Simulate movement based on the DYNAMIC hold_time
#             distance_to_move = game_config['max_speed_pps'] * action['hold_time']
#             move_vector = Vector2D()
#             if "left" in action['key_press']: move_vector.x -= 1
#             if "right" in action['key_press']: move_vector.x += 1
#             if "up" in action['key_press']: move_vector.y -= 1
#             if "down" in action['key_press']: move_vector.y += 1
            
#             print(move_vector.normalize(), distance_to_move, move_vector.x)
#             character_position += move_vector.normalize() * distance_to_move
#         else:
#             # This will now trigger when we are in range
#             print("\n--- ACTION IS NONE: Target likely reached or no movement needed. ---\n")
#             break

#     final_distance = (character_position - target_position).magnitude()
#     print(f"Simulation ended. Final distance to target: {final_distance:.2f} pixels.")



# if __name__ == '__main__':
#     # --- 1. Define Game Configuration ---
#     # This dictionary mimics knowing the rules of the game you're playing.
#     game_config_8_way = {
#         'movement_mode': 'acceleration',
#         'directional_constraints': 'eight_way',
#         'max_speed': 4.0
#     }
    
#     game_config_4_way_const = {
#         'movement_mode': 'constant_speed',
#         'directional_constraints': 'four_way',
#         'max_speed': 3.0 # A constant speed game
#     }

#     # --- 2. Initialize Controller and Game State ---
#     print("--- SIMULATION 1: 8-Way Acceleration Movement ---")
#     controller = MovementController(game_config_8_way)
    
#     # Mock game state
#     character_position = Vector2D(499, 436)
#     target_position = Vector2D(359, 640)
#     obstacles = [
#         {'position': Vector2D(400, 280), 'radius': 50, 'impenetrable': True},
#         {'position': Vector2D(400, 380), 'radius': 50, 'impenetrable': False} # A non-solid obstacle
#     ]

#     # --- 3. Run The Simulation Loop ---
#     time.sleep(5)
#     for i in range(250):
#         # VLM would provide this data in a real scenario
#         action = controller.decide_action(character_position, target_position, obstacles)
        
#         # --- Mock Game Engine ---
#         # A real game would execute the action. We'll simulate it.
#         print(f"Tick {i+1:03d} | Pos: {character_position} | Action: {action}")
        
#         if action:
#             # Simple simulation: move 3 pixels in the direction of the keys
#             if "left" in action['key_press']: character_position.x -= 36
#             if "right" in action['key_press']: character_position.x += 36
#             if "up" in action['key_press']: character_position.y -= 36
#             if "down" in action['key_press']: character_position.y += 36
#             if "left" in action['key_press']: take_action(key_press=action["key_press"])
#             if "right" in action['key_press']: take_action(key_press=action["key_press"])
#             if "up" in action['key_press']: take_action(key_press=action["key_press"])
#             if "down" in action['key_press']: take_action(key_press=action["key_press"])
#         # ------------------------

#         if (character_position - target_position).magnitude() < 10:
#             print("\n--- TARGET REACHED! ---\n")
#             break

# # --- 2. The Agent (The "Pilot") ---
# # This class contains all the logic for movement and decision-making.
# class Agent:
#     """Represents the character, handling its own physics and steering."""
#     def __init__(self, start_pos_x, start_pos_y):
#         self.position = Vector2D(start_pos_x, start_pos_y)
#         self.velocity = Vector2D(0, 0)
#         self.acceleration = Vector2D(0, 0)
        
#         # --- Configurable Parameters ---
#         self.max_speed = 4.0        # How fast the agent can move
#         self.max_force = 0.15       # How sharply the agent can turn
#         self.slowing_radius = 100.0 # The radius to start slowing down for arrival
#         self.avoidance_radius = 50.0 # The radius to "see" obstacles
        
#         # Behavior weights
#         self.arrive_weight = 0.5
#         self.avoidance_weight = 2.0

#     def apply_force(self, force):
#         """Adds a force to the agent's acceleration for the current tick."""
#         self.acceleration += force

#     def update(self):
#         """Updates the agent's position based on its physics."""
#         self.velocity += self.acceleration
#         self.velocity = self.velocity.limit(self.max_speed)
#         self.position += self.velocity
#         self.acceleration *= 0  # Reset acceleration for the next frame/tick

#     # --- Steering Behavior Logic ---

#     def _arrive(self, target_pos):
#         """Calculates the steering force to arrive at a target."""
#         desired_velocity = target_pos - self.position
#         distance = desired_velocity.magnitude()

#         if distance < self.slowing_radius:
#             desired_speed = (distance / self.slowing_radius) * self.max_speed
#             desired_velocity = desired_velocity.set_magnitude(desired_speed)
#         else:
#             desired_velocity = desired_velocity.set_magnitude(self.max_speed)

#         steering_force = desired_velocity - self.velocity
#         return steering_force.limit(self.max_force)

#     def _obstacle_avoidance(self, obstacles):
#         """Calculates the steering force to avoid a list of obstacles."""
#         total_avoidance_force = Vector2D(0, 0)

#         #matrixify?
#         for obstacle in obstacles:
#             dist_to_obstacle = (obstacle['position'] - self.position).magnitude()
            
#             # Combine agent and obstacle radius for collision check
#             effective_radius = self.avoidance_radius + obstacle['radius']

#             if 0 < dist_to_obstacle < effective_radius:
#                 flee_force = self.position - obstacle['position']
#                 # Scale force: the closer the obstacle, the stronger the repulsion
#                 scale = 1 - (dist_to_obstacle / effective_radius)
#                 flee_force = flee_force.set_magnitude(self.max_force * scale)
#                 total_avoidance_force += flee_force
        
#         # Avoidance can be a stronger, more urgent force
#         return total_avoidance_force.limit(self.max_force * self.avoidance_weight)


#     def compute_steering(self, target_pos, obstacles):
#         """
#         Calculates all steering forces, combines them, and applies them.
#         This is the "thinking" part of the agent for a single tick.
#         """
#         arrive_force = self._arrive(target_pos) * self.arrive_weight
#         avoidance_force = self._obstacle_avoidance(obstacles)

#         # Apply the combined forces
#         self.apply_force(arrive_force)
#         self.apply_force(avoidance_force)


# # --- 3. The Main Execution Logic ---
# # This is where you would integrate the VLM data.
# if __name__ == '__main__':
#     print("--- Running Steering System Simulation (Functional Skeleton) ---")

#     # 1. INITIALIZE THE AGENT
#     # Create an agent starting at position (100, 100)
#     my_agent = Agent(start_pos_x=499, start_pos_y=436)
#     print(f"Initial Agent Position: {my_agent.position}")

#     # 2. DEFINE THE ENVIRONMENT (This is where VLM input goes)
#     # The VLM would provide these coordinates in a real application.
#     target_position = Vector2D(359, 640)
    
#     # Obstacles are represented as a list of dictionaries.
#     # Each dict has a 'position' (Vector2D) and a 'radius' (float).
#     obstacle_list = [
#         {'position': Vector2D(350, 250), 'radius': 50},
#         {'position': Vector2D(400, 400), 'radius': 80},
#         {'position': Vector2D(600, 300), 'radius': 40},
#     ]

#     print(f"Target set to: {target_position}")
#     print(f"Obstacles loaded: {len(obstacle_list)}")
#     print("-" * 20)

#     # 3. RUN THE SIMULATION LOOP
#     # We simulate 150 frames/ticks of movement.
#     simulation_steps = 150
#     for i in range(simulation_steps):
#         # --- This is the core loop for your application ---
        
#         # A. (VLM STEP) Get the latest data. For this simulation, data is static.
#         # In a real app: target_position, obstacle_list = get_vlm_data()
        
#         # B. (AGENT THINKING) Agent calculates its desired movement for this tick.
#         my_agent.compute_steering(target_position, obstacle_list)
        
#         # C. (AGENT ACTION) Agent updates its position based on the calculated forces.
#         my_agent.update()
        
#         # D. (OUTPUT) Print the agent's new position.
#         print(f"Step {i+1:03d}: Agent Position = {my_agent.position}")

#         # Check for arrival (optional)
#         if (my_agent.position - target_position).magnitude() < 5:
#             print("\n--- Target Reached! ---")
#             break
            
#     print("\n--- Simulation Complete ---")