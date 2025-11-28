from pydantic import BaseModel, Field, field_validator, AfterValidator
from interfaces.vllm_interface import JohnVLLM
from interfaces.base import BaseModelConfig
from typing import List, Literal, Annotated, Optional
from PIL import Image
import json

#Paper Lily -- only a testing temp one
# control_schema ={
#     "move_up": "moves the player forward or moves selection up",
#     "move_down": "moves the player down/backwards or moves selection down",
#     "move_left": "moves the player left or moves selection left",
#     "move_right": "moves the player right or moves selection right",
#     "interact": "interacts with the environment",
#     #add combat related values.

# }

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
    
model_init_kwargs = {"gpu_memory_utilization": 0.93, "max_model_len": 8000, "trust_remote_code": True,
    }
model_config = BaseModelConfig(model_path_or_id="Qwen/Qwen3-VL-8B-Instruct-FP8", is_vision_model=True, uses_special_chat_template=False, model_init_kwargs=model_init_kwargs)
llm = JohnVLLM(model_config).load_model(model_config)


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
# import time
# import os
# import logging
# import cv2
# import numpy as np
# from typing import Optional

# # --- Imports ---
# # 1. The Optimized Capture System (saved from previous step)
# from gaming.game_capture import CaptureManager, SystemConfig

# # 2. Your Input Controller
# try:
#     from gaming.controls import InputControllerThread
# except ImportError:
#     # Mocking for demonstration if the file isn't present locally
#     import threading
#     class InputControllerThread(threading.Thread):
#         def execute_action(self, action): print(f"[MockInput] Executing: {action}")
#         def stop(self): pass
#         def run(self): pass

# # --- Setup Logging ---
# logging.basicConfig(
#     level=logging.INFO, 
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%H:%M:%S"
# )


# # hacky temporary fix for running with sudo -- gives permission of captured images to user.
# def save_as_user(path, image):
#     """Saves an image and immediately restores ownership to the non-root user."""
#     # 1. Save the file (currently owned by root)
#     cv2.imwrite(path, image)
    
#     # 2. Check if we are running via sudo
#     sudo_uid = os.environ.get('SUDO_UID')
#     sudo_gid = os.environ.get('SUDO_GID')
    
#     if sudo_uid and sudo_gid:
#         try:
#             # 3. Change ownership back to the original user
#             os.chown(path, int(sudo_uid), int(sudo_gid))
#         except Exception as e:
#             logging.warning(f"Could not change file ownership: {e}")


# def capture_action_sequence():
#     """
#     Orchestrates the Before -> Action -> After capture flow.
#     """
#     # --- Configuration ---
#     ACTION_DURATION = 1.0
#     OUTPUT_DIR = 'debug_frames/character_dataset'
    
#     # Configure the system for VLM (1000x1000)
#     config = SystemConfig(
#         device_index=0,
#         src_width=2560,
#         src_height=1440,
#         target_fps=60,
#         target_size=(1000, 1000), # VLM Standard
#         enable_psutil=True,
#         warmup_time=2.0
#     )

#     manager = CaptureManager(config)
#     input_controller = InputControllerThread()
    
#     try:
#         # 1. Start Background Processes (Includes Warmup)
#         logging.info("System: Initializing workers...")
#         input_controller.start()
#         manager.start_system() # This blocks for 2.0s for warmup
        
#         time.sleep(2.0)
#         # 2. Capture "Before" State
#         logging.info("Phase: Capturing 'Before' frame...")
#         # We capture a tiny slice of time to ensure we get a fresh frame
#         before_frame_raw = manager.get_snapshot()
#         if before_frame_raw is None:
#             raise RuntimeError("Failed to capture 'Before' frame.")

#         # 3. Capture "Action" State
#         logging.info(f"Phase: Executing Action for {ACTION_DURATION}s...")
        
#         # Start filling the RAM buffer
#         manager.start_capture()
        
#         # Trigger the physical action
#         start_t = time.perf_counter()
#         input_controller.execute_action({
#             "type": "key_press",
#             "details": {"key": ["left"], "hold_time": ACTION_DURATION}
#         })
        
#         # Wait strictly for the duration
#         # We calculate sleep to ensure exact timing, accounting for execution overhead
#         elapsed = time.perf_counter() - start_t
#         remaining = ACTION_DURATION - elapsed
#         if remaining > 0:
#             time.sleep(remaining)
            
#         # Stop filling buffer
#         during_frames_raw = manager.stop_capture()
#         logging.info(f"Action captured: {len(during_frames_raw)} raw frames.")

#         # 4. Capture "After" State
#         # Wait a moment for physics/animations to settle
#         time.sleep(0.2)
        
#         logging.info("Phase: Capturing 'After' frame...")
#         after_frame_raw = manager.get_snapshot()
#         if after_frame_raw is None:
#             raise RuntimeError("Failed to capture 'After' frame.")

#         # 5. Post-Processing (The Heavy Lifting)
#         logging.info("Phase: Post-Processing (Resizing & Letterboxing)...")
        
#         # Combine everything into one list to maximize thread pool efficiency
#         # Structure: [Before] + [During...] + [After]
#         all_raw_frames = [before_frame_raw] + during_frames_raw + [after_frame_raw]
        
#         t0 = time.perf_counter()
#         # This runs the 1000x1000 letterbox logic on all cores
#         all_processed = manager.post_process_frames(all_raw_frames)
#         logging.info(f"Processed {len(all_processed)} frames in {time.perf_counter() - t0:.3f}s")

#         # Separate them back out
#         before_final = all_processed[0]
#         during_final = all_processed[1:-1]
#         after_final = all_processed[-1]

#         # 6. Save to Disk
#         # Note: cv2.imwrite expects BGR, which is what we have. No conversion needed.
#         logging.info(f"Saving to {OUTPUT_DIR}...")
#         os.makedirs(OUTPUT_DIR, exist_ok=True)
        
#         sudo_uid = os.environ.get('SUDO_UID')
#         sudo_gid = os.environ.get('SUDO_GID')
#         if sudo_uid and sudo_gid:
#             os.chown(OUTPUT_DIR, int(sudo_uid), int(sudo_gid))

#         # Use the helper function instead of cv2.imwrite directly -- TODO: drop when moving on.
#         save_as_user(os.path.join(OUTPUT_DIR, 'capture_before_action.png'), before_final)
#         save_as_user(os.path.join(OUTPUT_DIR, 'capture_after_action.png'), after_final)
        
#         for i, frame in enumerate(during_final):
#             fname = f"during_action_{i:04d}.png"
#             save_as_user(os.path.join(OUTPUT_DIR, fname), frame)
#         logging.info("Sequence complete.")

#     except Exception as e:
#         logging.error(f"Critical Error: {e}", exc_info=True)
#     finally:
#         # Clean shutdown
#         if input_controller.is_alive():
#             input_controller.stop()
#         manager.stop_system()
#         logging.info("System shutdown.")

# if __name__ == "__main__":
#     capture_action_sequence()



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













































#async vllm generator test
# from interfaces.vllm_interface import JohnVLLMAsync
# from interfaces.base import BaseModelConfig


# generation_config = {
#     "max_tokens": 512,
#     "temperature": 0.7,
#     "top_p": 1.0,
#     "top_k": -1,
#     "repetition_penalty": 1.0,
#     "output_kind": "DELTA",
#     # Guided decoding or other complex params can be added here
# }

# import asyncio

# model_init_kwargs = {"gpu_memory_utilization": 0.93, "max_model_len": 8000, "trust_remote_code": True,
#     }
# model_config = BaseModelConfig(model_path_or_id="Qwen/Qwen3-VL-8B-Instruct-FP8", is_vision_model=True, uses_special_chat_template=False, model_init_kwargs=model_init_kwargs)

# async def generate():
#     vllm = await JohnVLLMAsync(model_config).load_model(model_config)
#     # async with vllm:

#     async for response in vllm.dialogue_generator(prompt="Hello", generation_config=generation_config):
#         print(response)


# asyncio.run(generate())
