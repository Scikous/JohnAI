# from __future__ import annotations
# import asyncio
# import json
# import logging
# import multiprocessing as mp
# import threading
# from collections import deque
# from typing import Literal, Any
# from enum import Enum, auto

# from pynput import keyboard
# from pynput.keyboard import Key, KeyCode
# from model_utils import load_character

# from PIL import Image
# from LLM_Wizard.gaming.game_capture import GameCaptureWorker
# from LLM_Wizard.gaming.controls import InputController

# from interfaces.vllm_interface import JohnVLLM
# from interfaces.base import BaseModelConfig
# from pydantic import BaseModel, Field, ValidationError

# # --- CONFIGURATION AND SETUP ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Character and Model Configuration
# character_info_json = "LLM_Wizard/characters/character.json"
# instructions, user_name, character_name = load_character(character_info_json)

# # --- AGENT STATE DEFINITION ---
# class AgentState(Enum):
#     """Defines the current high-level state of the gaming agent."""
#     INITIALIZING = auto()
#     MAPPING_CONTROLS = auto()
#     REMAPPING_CONTROL = auto()
#     PLANNING = auto()
#     EXECUTING = auto()
#     COMBAT = auto()

# # --- PYDANTIC MODELS FOR STRUCTURED I/O ---

# class Desire(BaseModel):
#     """A high-level objective determined by the Planner."""
#     desire: str = Field(..., description="A concise, semantic description of the goal, e.g., 'navigate_to_quest_marker', 'open_inventory', 'win_combat'.")
#     details: dict[str, Any] | None = Field(None, description="Optional dictionary for additional context, like coordinates or target descriptions.")

# class DiscoveredControl(BaseModel):
#     """Represents a single, verified game control mapping."""
#     action_name: str = Field(..., description="The semantic name for the action, e.g., 'confirm_selection', 'move_forward'.")
#     input_details: dict[str, Any] = Field(..., description="The low-level input required, e.g., {'type': 'key_press', 'key': 'e'}.")

# class ActionOutcome(BaseModel):
#     """Represents the result of a single action or a completed task."""
#     success: bool
#     reason: str
#     missing_control_for_desire: str | None = Field(None, description="If the action failed because the control is unknown, this holds the desire that could not be fulfilled.")

# class LowLevelAction(BaseModel):
#     """A single, concrete action to be executed by the InputController."""
#     type: Literal["key_press", "mouse_click", "mouse_move"]
#     details: dict[str, Any]

# # --- PROMPT TEMPLATES (FULL & REVISED) ---

# PLANNER_PROMPT_TEMPLATE = """
# You are a master game strategist. Your task is to determine the next high-level objective.
# Analyze the provided `game_screenshot` and the current `world_model`.
# Your response MUST be a single JSON object conforming to the Desire schema.

# **World Model (Current Knowledge):**
# {world_model_json}

# **Your Goal:** Decide the single most important next step.
# Examples: "start_new_game", "explore_vicinity", "read_quest_log", "engage_enemy".

# **Output Schema:**
# {{
#   "desire": "<your_concise_desire>",
#   "details": {{ "key": "value" }} // Optional
# }}

# Your response must be ONLY the JSON object.
# """

# MAPPER_CHANGE_DETECT_PROMPT_TEMPLATE = """
# You are a visual analysis AI. Your task is to determine if a meaningful change occurred between two images after a key press.
# A key '{key_pressed}' was pressed. Analyze the `before_image` and `after_image`.
# Meaningful changes include: a menu appearing/disappearing, character moving, item being selected, door opening.
# Ignore subtle changes like environmental animations (e.g., leaves rustling).

# Your response MUST be a single JSON object.
# **Format:**
# {{
#   "change_detected": <true_or_false>,
#   "description": "<A brief description of the change if detected, otherwise empty.>"
# }}
# """

# MAPPER_NAME_ACTION_PROMPT_TEMPLATE = """
# You are a game analysis AI. A specific key press ('{key_pressed}') caused a described visual change.
# Your task is to give this action a concise, semantic name based on the change.
# Examples:
# - Change: "The inventory screen appeared." -> Name: "open_inventory"
# - Change: "The character jumped." -> Name: "jump"
# - Change: "The selected menu item moved down." -> Name: "navigate_menu_down"

# **Visual Change Description:**
# {change_description}

# Your response MUST be a single JSON object.
# **Format:**
# {{
#   "action_name": "<your_semantic_action_name>"
# }}
# """

# PLAYER_EXECUTE_PROMPT_TEMPLATE = """
# You are a tactical game-playing AI. Your goal is to execute a high-level desire by selecting the correct low-level action from the KNOWN controls.
# Analyze the `game_screenshot` and use the `control_schema` to achieve the `current_desire`.

# **CRITICAL INSTRUCTION:** Review the `input_modes`. If `mouse_supported` is `false`, you are FORBIDDEN from choosing `mouse_click` or `mouse_move` actions. You MUST use a `key_press`.

# **Current Desire:**
# {desire_json}

# **Input Modes:**
# {input_modes_json}

# **Known Controls (Control Schema):**
# {control_schema_json}

# Your task is to choose the single best input from the KNOWN controls to progress the desire.
# Your response MUST be a single JSON object conforming to the LowLevelAction schema.
# **Format:**
# {{
#   "type": "<'key_press' or 'mouse_click'>",
#   "details": {{...}}
# }}
# """

# PLAYER_VERIFY_PROMPT_TEMPLATE = """
# You are a visual verification AI. An action was just performed to achieve a specific desire.
# Analyze the `game_screenshot` to determine if the desire was successfully achieved.

# **Original Desire:**
# {desire_json}

# **Action Performed:**
# {action_performed_json}

# Your response MUST be a single JSON object.
# **Format:**
# {{
#   "success": <true_or_false>,
#   "reason": "<Briefly explain why it succeeded or failed based on the visual evidence.>"
# }}
# """

# # --- HELPER FUNCTIONS ---
# def panic_switch_listener(stop_event: mp.Event):
#     """Listens for Ctrl+Alt+Q to terminate the agent."""
#     hotkey = {Key.ctrl, Key.alt, KeyCode.from_char('q')}
#     current_keys = set()
#     logging.info("Panic switch active. Press Ctrl+Alt+Q to terminate.")
#     def on_press(key):
#         if key in hotkey:
#             current_keys.add(key)
#             if all(k in current_keys for k in hotkey):
#                 logging.warning("PANIC SWITCH HOTKEY DETECTED! Shutting down.")
#                 stop_event.set()
#                 listener.stop()
#     def on_release(key):
#         try: current_keys.remove(key)
#         except KeyError: pass
#     with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
#         listener.join()

# async def execute_action(action: LowLevelAction, controller: InputController):
#     """Executes a game action using the InputController."""
#     if not action:
#         logging.warning("execute_action received an empty action.")
#         return
#     await asyncio.to_thread(controller.execute_action, action.model_dump())

# def get_latest_frame(image_queue: mp.Queue, images_deque: deque) -> Image.Image | None:
#     """Drains the queue and returns the most recent frame."""
#     while not image_queue.empty():
#         data_packet = image_queue.get_nowait()
#         images_deque.append(Image.frombytes(data_packet['mode'], data_packet['size'], data_packet['image_bytes']))
#     return images_deque[-1] if images_deque else None

# # --- SPECIALIZED AGENT CLASSES ---

# class PlannerAgent:
#     """The Strategist: Decides the high-level objective."""
#     def __init__(self, llm: JohnVLLM):
#         self.llm = llm
#         self.gen_config = {"max_tokens": 256, "temperature": 0.2}

#     async def decide_next_desire(self, latest_frame: Image.Image, world_model: dict) -> Desire | None:
#         logging.info("--- PLANNER: Deciding next desire... ---")
#         prompt = PLANNER_PROMPT_TEMPLATE.format(world_model_json=json.dumps(world_model, indent=2))
#         assistant_prompt = '{\n  "desire": "'
        
#         output = self.llm.dialogue_generator(
#             prompt=prompt, assistant_prompt=assistant_prompt, images=[latest_frame],
#             add_generation_prompt=False, continue_final_message=True, generation_config=self.gen_config
#         )
#         try:
#             parsed_response = Desire.model_validate_json(assistant_prompt + output)
#             logging.info(f"Planner decided on new desire: {parsed_response.desire}")
#             return parsed_response
#         except (ValidationError, json.JSONDecodeError) as e:
#             logging.error(f"Planner failed to generate a valid desire: {e}\nRaw output: {output}")
#             return None

# class MapperAgent:
#     """The Cartographer: Discovers and maps game controls, now with targeted discovery."""
#     def __init__(self, llm: JohnVLLM, controller: InputController):
#         self.llm = llm
#         self.controller = controller
#         self.gen_config = {"max_tokens": 128, "temperature": 0.1}
#         self.candidate_keys = [
#             'e', 'f', 'i', 'm', 'tab', 'escape', 'space', 'enter', 'c', 'j', 'q', 'r', 'x', 'z',
#             'w', 'a', 's', 'd', 'up', 'down', 'left', 'right', 'shift', 'ctrl', 'alt'
#         ]

#     async def suggest_keys(self, latest_frame: Image.Image, desire: str, candidates: list[str], known_keys: set | None = None, world_model: dict | None = None) -> list[str]:
#         suggest_prompt = f"""
#         Analyze the game screenshot and world model to suggest 3-5 keys from the candidates most likely to achieve '{desire}'.
#         Candidates: {', '.join(candidates)}
#         Prioritize based on common game conventions (e.g., 'e' for interact, 'esc' for menus).
#         AVOID keys that might cause irreversible harm based on the screenshot:
#         - Movement (w/a/s/d) if near edges/cliffs/danger.
#         - 'esc' or 'q' if in menus with 'Quit' options.
#         - Any key highlighting destructive UI elements.
#         If no safe keys, suggest an empty list.
#         """
#         if known_keys:
#             suggest_prompt += f"\nExclude these already known keys: {', '.join(known_keys)}"
#         if world_model and world_model.get("input_modes", {}).get("mouse_supported"):
#             suggest_prompt += "\nInclude mouse actions like 'mouse_left_click', 'mouse_right_click' if relevant."
#         suggest_prompt += "\nOutput ONLY a JSON list: [\"key1\", \"key2\", ...]"
#         assistant_prompt = '['
#         output = self.llm.dialogue_generator(
#             prompt=suggest_prompt, assistant_prompt=assistant_prompt, images=[latest_frame],
#             add_generation_prompt=False, continue_final_message=True, generation_config={"max_tokens": 128, "temperature": 0.05}
#         )
#         try:
#             suggested = json.loads(assistant_prompt + output)
#             logging.info(f"Suggested keys for '{desire}': {suggested}")
#             return suggested
#         except json.JSONDecodeError as e:
#             logging.error(f"Failed to parse suggested keys: {e}")
#             return []

#     async def _test_key(self, key: str, image_queue: mp.Queue, images_deque: deque, desire: str) -> tuple[str, dict] | None:
#         """Core logic to test a single key and return its mapping if a change is detected."""
#         logging.info(f"Mapper: Testing key '{key}'...")
#         if key is not str:
#             key = key[0]
#         before_frame = get_latest_frame(image_queue, images_deque)
#         if not before_frame:
#             logging.warning(f"Mapper: Could not get 'before' frame for key '{key}'.")
#             return None

#         # Pre-test safety check
#         predict_prompt = f"Would pressing '{key}' in this screenshot likely cause irreversible damage (e.g., quit, death)? Yes/no and brief reason."
#         predict_assistant_prompt = ""
#         predict_output = self.llm.dialogue_generator(
#             prompt=predict_prompt, assistant_prompt=predict_assistant_prompt, images=[before_frame],
#             add_generation_prompt=False, continue_final_message=False, generation_config={"max_tokens": 20, "temperature": 0.05}
#         )
#         if "yes" in predict_output.lower():
#             logging.warning(f"Skipping '{key}' as potentially destructive: {predict_output}")
#             return None

#         # Determine action type
#         if 'mouse_' in key:
#             button = key.split('_')[-2] if '_click' in key else None
#             action = LowLevelAction(type="mouse_click", details={"button": button} if button else {"position": "center"})  # Assume default position
#         else:
#             action = LowLevelAction(type="key_press", details={"key": key})

#         await execute_action(action, self.controller)
#         await asyncio.sleep(1.0)

#         after_frame = get_latest_frame(image_queue, images_deque)
#         if not after_frame:
#             logging.warning(f"Mapper: Could not get 'after' frame for key '{key}'.")
#             return None

#         # Step 1: Detect if a change occurred
#         prompt = MAPPER_CHANGE_DETECT_PROMPT_TEMPLATE.format(key_pressed=key)
#         assistant_prompt = '{\n  "change_detected": '
#         output = self.llm.dialogue_generator(
#             prompt=prompt, assistant_prompt=assistant_prompt, images=[before_frame, after_frame],
#             add_generation_prompt=False, continue_final_message=True, generation_config=self.gen_config
#         )
#         try:
#             change_result = json.loads(assistant_prompt + output)
#             if change_result.get("change_detected"):
#                 logging.info(f"Mapper: Detected change for key '{key}'. Description: {change_result.get('description')}")
                
#                 # Step 2: Name the detected action
#                 name_prompt = MAPPER_NAME_ACTION_PROMPT_TEMPLATE.format(
#                     key_pressed=key, change_description=change_result.get('description')
#                 )
#                 name_assistant_prompt = '{\n  "action_name": "'
#                 name_output = self.llm.dialogue_generator(
#                     prompt=name_prompt, assistant_prompt=name_assistant_prompt, images=None,
#                     add_generation_prompt=False, continue_final_message=True, generation_config=self.gen_config
#                 )
#                 name_result = json.loads(name_assistant_prompt + name_output)
#                 action_name = name_result["action_name"]
                
#                 # Step 3: Validate relevance to desire
#                 validate_prompt = f"Does the action '{action_name}' achieve the desire '{desire}' in the current game context? Respond with only 'yes' or 'no'."
#                 validate_output = self.llm.dialogue_generator(
#                     prompt=validate_prompt, assistant_prompt="", images=[after_frame],
#                     add_generation_prompt=False, continue_final_message=False, generation_config={"max_tokens": 5, "temperature": 0.05}
#                 )
#                 if "yes" in validate_output.lower():
#                     input_details = action.details  # Use the details from the action
#                     # Press escape to close any opened menus to reset state for next test
#                     await execute_action(LowLevelAction(type="key_press", details={"key": "escape"}), self.controller)
#                     await asyncio.sleep(1.0)
#                     return action_name, input_details
#         except (ValidationError, json.JSONDecodeError) as e:
#             logging.error(f"Mapper failed to parse response for key '{key}': {e}")
        
#         return None

#     async def discover_controls(self, image_queue: mp.Queue, images_deque: deque, known_keys: set, world_model: dict) -> dict:
#         """Performs a broad scan of candidate keys not already known."""
#         logging.info("--- MAPPER: Starting initial broad control discovery... ---")
#         latest_frame = get_latest_frame(image_queue, images_deque)
#         if not latest_frame:
#             return {}
#         suggested = await self.suggest_keys(latest_frame, "discover new controls safely", self.candidate_keys, known_keys=known_keys, world_model=world_model)
#         control_schema = {}
#         for key in suggested:
#             mapping = await self._test_key(key, image_queue, images_deque, "broad_discovery")  # Use a placeholder desire for broad
#             if mapping:
#                 action_name, input_details = mapping
#                 control_schema[action_name] = input_details
#         logging.info(f"--- MAPPER: Broad discovery complete. Found {len(control_schema)} new controls. ---")
#         return control_schema

#     async def discover_specific_control(self, desire_to_map: str, image_queue: mp.Queue, images_deque: deque, known_keys: set, world_model: dict) -> tuple[str, dict] | None:
#         """Performs a targeted scan to find a control for a specific desire."""
#         logging.info(f"--- MAPPER: Starting targeted search for desire: '{desire_to_map}' ---")
#         latest_frame = get_latest_frame(image_queue, images_deque)
#         if not latest_frame:
#             return None
#         suggested = await self.suggest_keys(latest_frame, desire_to_map, self.candidate_keys, world_model=world_model)  # No exclusion for aliases
#         for key in suggested:
#             mapping = await self._test_key(key, image_queue, images_deque, desire_to_map)
#             if mapping:
#                 action_name, input_details = mapping
#                 logging.info(f"Mapper: Found direct semantic match for '{desire_to_map}': action '{action_name}' with key '{key}'")
#                 return action_name, input_details
#         logging.warning(f"Mapper: Targeted search for '{desire_to_map}' failed to find any new control.")
#         return None

# class PlayerAgent:
#     """The Hands: Executes desires, now signals when a control is unknown."""
#     def __init__(self, llm: JohnVLLM, controller: InputController):
#         self.llm = llm
#         self.controller = controller
#         self.gen_config = {"max_tokens": 256, "temperature": 0.1}

#     async def execute_desire(self, desire: Desire, world_model: dict, control_schema: dict, image_queue: mp.Queue, images_deque: deque) -> ActionOutcome:
#         logging.info(f"--- PLAYER: Executing desire: {desire.desire} ---")
        
#         if not any(desire.desire.lower().replace("_", "") in key.lower().replace("_", "") for key in control_schema.keys()):
#             logging.warning(f"Player: No known control seems to match desire '{desire.desire}'. Requesting remap.")
#             return ActionOutcome(success=False, reason="Control for desire is not in the known schema.", missing_control_for_desire=desire.desire)

#         # Replay known sequence if available
#         known_sequences = world_model.setdefault("known_sequences", {})
#         known_seq = known_sequences.get(desire.desire, [])
#         if known_seq:
#             logging.info(f"Player: Replaying known sequence for '{desire.desire}'")
#             for act_dict in known_seq:
#                 action = LowLevelAction.model_validate(act_dict)
#                 await execute_action(action, self.controller)
#                 await asyncio.sleep(1.5)
#             after_frame = get_latest_frame(image_queue, images_deque)
#             if not after_frame:
#                 logging.warning("Player: Could not get frame for verification after sequence replay. Assuming failure.")
#                 return ActionOutcome(success=False, reason="Failed to get verification frame after sequence.")
#             verify_prompt = PLAYER_VERIFY_PROMPT_TEMPLATE.format(
#                 desire_json=desire.model_dump_json(indent=2),
#                 action_performed_json=json.dumps(known_seq, indent=2)  # Whole sequence as 'action'
#             )
#             verify_assistant_prompt = '{\n  "success": '
#             verify_output = self.llm.dialogue_generator(
#                 prompt=verify_prompt, assistant_prompt=verify_assistant_prompt, images=[after_frame],
#                 add_generation_prompt=False, continue_final_message=True, generation_config=self.gen_config
#             )
#             try:
#                 outcome = ActionOutcome.model_validate_json(verify_assistant_prompt + verify_output)
#                 if outcome.success:
#                     logging.info(f"Player: Successfully executed desire '{desire.desire}' via sequence. Reason: {outcome.reason}")
#                     return outcome
#                 else:
#                     logging.warning(f"Known sequence failed for '{desire.desire}'. Reason: {outcome.reason}. Falling back to dynamic execution.")
#             except (ValidationError, json.JSONDecodeError) as e:
#                 logging.error(f"Player failed to parse verification response for sequence: {e}")

#         max_attempts = 3
#         actions_performed: list[dict] = []
#         for attempt in range(max_attempts):
#             latest_frame = get_latest_frame(image_queue, images_deque)
#             if not latest_frame:
#                 return ActionOutcome(success=False, reason="Could not get game frame.")

#             prompt = PLAYER_EXECUTE_PROMPT_TEMPLATE.format(
#                 desire_json=desire.model_dump_json(indent=2),
#                 input_modes_json=json.dumps(world_model.get("input_modes", {})),
#                 control_schema_json=json.dumps(control_schema, indent=2)
#             )
#             assistant_prompt = '{\n  "type": "'
#             output = self.llm.dialogue_generator(
#                 prompt=prompt, assistant_prompt=assistant_prompt, images=[latest_frame],
#                 add_generation_prompt=False, continue_final_message=True, generation_config=self.gen_config
#             )
#             try:
#                 action_to_perform = LowLevelAction.model_validate_json(assistant_prompt + output)
#                 logging.info(f"Player: Attempt {attempt+1}/{max_attempts}, performing action: {action_to_perform.model_dump()}")
#             except (ValidationError, json.JSONDecodeError) as e:
#                 logging.error(f"Player failed to generate a valid action: {e}")
#                 await asyncio.sleep(2)
#                 continue

#             await execute_action(action_to_perform, self.controller)
#             await asyncio.sleep(1.5)
#             actions_performed.append(action_to_perform.model_dump())

#             after_frame = get_latest_frame(image_queue, images_deque)
#             if not after_frame:
#                 logging.warning("Player: Could not get frame for verification. Assuming failure.")
#                 continue
            
#             verify_prompt = PLAYER_VERIFY_PROMPT_TEMPLATE.format(
#                 desire_json=desire.model_dump_json(indent=2),
#                 action_performed_json=action_to_perform.model_dump_json(indent=2)
#             )
#             verify_assistant_prompt = '{\n  "success": '
#             verify_output = self.llm.dialogue_generator(
#                 prompt=verify_prompt, assistant_prompt=verify_assistant_prompt, images=[after_frame],
#                 add_generation_prompt=False, continue_final_message=True, generation_config=self.gen_config
#             )
#             try:
#                 outcome = ActionOutcome.model_validate_json(verify_assistant_prompt + verify_output)
#                 if outcome.success:
#                     logging.info(f"Player: Successfully executed desire '{desire.desire}'. Reason: {outcome.reason}")
#                     if actions_performed:
#                         known_sequences[desire.desire] = actions_performed
#                         logging.info(f"Saved new sequence for '{desire.desire}' with {len(actions_performed)} steps.")
#                     return outcome
#                 else:
#                     logging.warning(f"Player: Action failed to achieve desire. Reason: {outcome.reason}. Retrying...")
#             except (ValidationError, json.JSONDecodeError) as e:
#                 logging.error(f"Player failed to parse verification response: {e}")
        
#         return ActionOutcome(success=False, reason=f"Failed to achieve desire '{desire.desire}' after {max_attempts} attempts.")

#     async def handle_combat(self, desire: Desire, control_schema: dict, image_queue: mp.Queue, images_deque: deque) -> ActionOutcome:
#         logging.info("--- PLAYER: Entering COMBAT mode ---")
#         attack_action_name = "attack"
#         if attack_action_name in control_schema:
#             details = control_schema[attack_action_name]
#             action = LowLevelAction(type=details['type'], details={"key": details['key']})
#             for _ in range(5):
#                 logging.info("Player (Combat): Attacking!")
#                 await execute_action(action, self.controller)
#                 await asyncio.sleep(1.0)
#             return ActionOutcome(success=True, reason="Executed pre-programmed combat sequence.")
#         else:
#             return ActionOutcome(success=False, reason="No 'attack' control mapped for combat.", missing_control_for_desire="attack")

# # --- MAIN AGENT LOGIC (HIGH-LEVEL CONTROLLER) ---

# async def run_gaming_agent():
#     """Main async function to run the modular, state-driven gaming agent."""
#     controller = InputController()
#     images_deque = deque(maxlen=10)
#     image_data_queue = mp.Queue()
#     command_queue = mp.Queue()
#     stop_event = mp.Event()

#     listener_thread = threading.Thread(target=panic_switch_listener, args=(stop_event,), daemon=True)
#     listener_thread.start()

#     capture_worker = GameCaptureWorker(
#         image_data_queue=image_data_queue, command_queue=command_queue, stop_event=stop_event,
#         interval_sec=0.2, source_type=1, target_size=(512, 384)
#     )

#     try:
#         capture_worker.start()
#         logging.info(f"Started capture worker with PID: {capture_worker.pid}")

#         model_init_kwargs = {"gpu_memory_utilization": 0.97, "max_model_len": 16384, "trust_remote_code": True}
#         model_config = BaseModelConfig(model_init_kwargs=model_init_kwargs)
#         llm = JohnVLLM(model_config).load_model(model_config)
#         logging.info("Model loaded. Initializing agents.")

#         planner = PlannerAgent(llm)
#         mapper = MapperAgent(llm, controller)
#         player = PlayerAgent(llm, controller)

#         agent_state = AgentState.INITIALIZING
#         world_model = {
#             "game_title": "Unknown",
#             "current_objective": "Initialize and detect input methods.",
#             "input_modes": {"mouse_supported": False},
#             "known_sequences": {}
#         }
#         control_schema = {}
#         current_desire: Desire | None = None
#         desire_to_remap: str | None = None

#         while not stop_event.is_set():
#             latest_frame = get_latest_frame(image_data_queue, images_deque)
#             if not latest_frame:
#                 logging.info("Waiting for the first game frame...")
#                 await asyncio.sleep(1)
#                 continue

#             if agent_state == AgentState.INITIALIZING:
#                 logging.info("State: INITIALIZING - Detecting mouse cursor...")
#                 prompt = "Does this game screen show a visible mouse cursor? Answer with only 'yes' or 'no'."
#                 output = llm.dialogue_generator(
#                     prompt=prompt, assistant_prompt="", images=[latest_frame],
#                     add_generation_prompt=False, continue_final_message=False, generation_config={"max_tokens": 5}
#                 )
#                 if "yes" in output.lower():
#                     world_model["input_modes"]["mouse_supported"] = True
#                     mapper.candidate_keys += ['mouse_left_click', 'mouse_right_click']
#                     logging.info("Mouse cursor detected. Mouse actions will be permitted and added to candidates.")
#                 else:
#                     world_model["input_modes"]["mouse_supported"] = False
#                     logging.info("No mouse cursor detected. Mouse actions will be forbidden.")
                
#                 logging.info("State: INITIALIZING -> MAPPING_CONTROLS")
#                 agent_state = AgentState.MAPPING_CONTROLS

#             elif agent_state == AgentState.MAPPING_CONTROLS:
#                 known_keys = {details.get('key') or details.get('button') for details in control_schema.values() if 'key' in details or 'button' in details}
#                 new_controls = await mapper.discover_controls(image_data_queue, images_deque, known_keys, world_model)
#                 control_schema.update(new_controls)
#                 world_model["current_objective"] = "Initial controls mapped. Determine next goal."
#                 logging.info("State: MAPPING_CONTROLS -> PLANNING")
#                 agent_state = AgentState.PLANNING

#             elif agent_state == AgentState.REMAPPING_CONTROL:
#                 if not desire_to_remap:
#                     logging.error("Entered REMAPPING state without a target desire. Returning to PLANNING.")
#                     agent_state = AgentState.PLANNING
#                     continue
                
#                 known_keys = {details.get('key') or details.get('button') for details in control_schema.values() if 'key' in details or 'button' in details}
#                 new_mapping = await mapper.discover_specific_control(desire_to_remap, image_data_queue, images_deque, known_keys, world_model)
#                 if new_mapping:
#                     action_name, input_details = new_mapping
#                     control_schema[action_name] = input_details
#                     logging.info(f"Successfully remapped and added new control: '{action_name}'")
#                 else:
#                     logging.warning(f"Failed to find a control for '{desire_to_remap}'. The agent may remain stuck on this task.")
                
#                 desire_to_remap = None
#                 logging.info("State: REMAPPING_CONTROL -> PLANNING")
#                 agent_state = AgentState.PLANNING

#             elif agent_state == AgentState.PLANNING:
#                 current_desire = await planner.decide_next_desire(latest_frame, world_model)
#                 if current_desire:
#                     if current_desire.desire == "win_combat":
#                         logging.info("State: PLANNING -> COMBAT")
#                         agent_state = AgentState.COMBAT
#                     else:
#                         logging.info("State: PLANNING -> EXECUTING")
#                         agent_state = AgentState.EXECUTING
#                 else:
#                     logging.warning("Planner failed to produce a desire. Retrying in 10 seconds.")
#                     await asyncio.sleep(10)

#             elif agent_state == AgentState.EXECUTING:
#                 if not current_desire:
#                     agent_state = AgentState.PLANNING
#                     continue
                
#                 outcome = await player.execute_desire(current_desire, world_model, control_schema, image_data_queue, images_deque)
#                 world_model["last_action_outcome"] = outcome.model_dump()
                
#                 if outcome.missing_control_for_desire:
#                     desire_to_remap = outcome.missing_control_for_desire
#                     logging.info(f"State: EXECUTING -> REMAPPING_CONTROL (Target: {desire_to_remap})")
#                     agent_state = AgentState.REMAPPING_CONTROL
#                 else:
#                     logging.info("State: EXECUTING -> PLANNING")
#                     agent_state = AgentState.PLANNING
#                 await asyncio.sleep(3)

#             elif agent_state == AgentState.COMBAT:
#                 if not current_desire:
#                     agent_state = AgentState.PLANNING
#                     continue

#                 outcome = await player.handle_combat(current_desire, control_schema, image_data_queue, images_deque)
#                 world_model["last_combat_outcome"] = outcome.model_dump()
                
#                 if outcome.missing_control_for_desire:
#                     desire_to_remap = outcome.missing_control_for_desire
#                     logging.info(f"State: COMBAT -> REMAPPING_CONTROL (Target: {desire_to_remap})")
#                     agent_state = AgentState.REMAPPING_CONTROL
#                 else:
#                     logging.info("State: COMBAT -> PLANNING")
#                     agent_state = AgentState.PLANNING
#                 await asyncio.sleep(3)

#     except Exception as e:
#         logging.error(f"An unhandled error occurred in the main loop: {e}", exc_info=True)
#     finally:
#         logging.info("Cleaning up...")
#         if 'capture_worker' in locals() and capture_worker.is_alive():
#             stop_event.set()
#             if controller: controller.close()
#             capture_worker.join(timeout=5)
#             if capture_worker.is_alive():
#                 logging.warning("Worker did not terminate gracefully, terminating.")
#                 capture_worker.terminate()
#         logging.info("Cleanup complete.")

# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#     asyncio.run(run_gaming_agent())



from __future__ import annotations
import asyncio
import json
import logging
import multiprocessing as mp
import threading
from collections import deque
from typing import Literal, Any, Deque
from enum import Enum, auto

from pynput import keyboard
from pynput.keyboard import Key, KeyCode
from model_utils import load_character

from PIL import Image
from LLM_Wizard.gaming.game_capture import GameCaptureWorker
from LLM_Wizard.gaming.controls import InputController

from interfaces.vllm_interface import JohnVLLM
from interfaces.base import BaseModelConfig
from pydantic import BaseModel, Field, ValidationError

# --- CONFIGURATION AND SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Character and Model Configuration
character_info_json = "LLM_Wizard/characters/character.json"
instructions, user_name, character_name = load_character(character_info_json)

# --- AGENT STATE DEFINITION (REVISED) ---
class AgentState(Enum):
    """Defines the current high-level state of the gaming agent's thinking loop."""
    DETERMINING_CONTEXT = auto()
    PLANNING = auto()
    EXECUTING = auto()
    DISCOVERING_CONTROL = auto()
    HANDLE_FAILURE = auto()

# --- PYDANTIC MODELS FOR STRUCTURED I/O (REVISED & EXPANDED) ---

class Desire(BaseModel):
    """A single, high-level objective. A sequence of these forms a Plan."""
    desire: str = Field(..., description="A concise, semantic description of the goal, e.g., 'navigate_to_quest_marker', 'open_inventory', 'attack_enemy'.")
    details: dict[str, Any] | None = Field(None, description="Optional dictionary for additional context, like coordinates or target descriptions.")

class Plan(BaseModel):
    """A sequence of Desires to achieve a larger goal."""
    thought: str = Field(..., description="A brief thought process explaining the reasoning behind the plan.")
    steps: list[Desire] = Field(..., description="The ordered list of desires to execute.")

class DiscoveredControl(BaseModel):
    """Represents a single, verified game control mapping."""
    action_name: str = Field(..., description="The semantic name for the action, e.g., 'confirm_selection', 'move_forward'.")
    input_details: dict[str, Any] = Field(..., description="The low-level input required, e.g., {'type': 'key_press', 'key': 'e'}.")

class ActionOutcome(BaseModel):
    """Represents the result of a single action or a completed task."""
    success: bool
    reason: str
    missing_control_for_desire: str | None = Field(None, description="If the action failed because the control is unknown, this holds the desire that could not be fulfilled.")

class LowLevelAction(BaseModel):
    """A single, concrete action to be executed by the InputController. Expanded for sequences."""
    type: Literal["key_press", "key_sequence", "mouse_click", "mouse_move"]
    details: dict[str, Any]

# --- PROMPT TEMPLATES (COMPLETELY REVISED) ---

CONTEXT_PROMPT_TEMPLATE = """
You are a game analysis AI. Analyze the provided `game_screenshot`.
Your task is to describe the current situation in the game concisely.
Focus on the game state, not just the visuals.
Examples: "main_menu", "in_game_world_exploring", "inventory_screen", "dialogue_with_npc", "combat_with_enemy".

Your response MUST be a single JSON object.
**Format:**
{{
  "context": "<your_concise_context_description>"
}}
"""

PLANNER_PROMPT_TEMPLATE = """
You are a master game strategist. Your task is to create a multi-step plan to make progress in the game.
Analyze the provided `game_screenshot` and the current `world_model`.

**World Model (Your Memory):**
{world_model_json}

**CRITICAL INSTRUCTIONS:**
1.  Review the `action_history`. If the last plan failed, you MUST devise a different strategy. Do not repeat failed actions.
2.  Your plan should be a logical sequence of small, achievable steps (Desires).
3.  Base your plan on the `current_context` and the available `control_schema`.
4.  If you are in a menu but want to be in the game world, your first step must be to exit the menu.

**Your Goal:** Create a plan to move forward.
Your response MUST be a single JSON object conforming to the Plan schema.

**Output Schema:**
{{
  "thought": "<Your reasoning for this plan>",
  "steps": [
    {{ "desire": "<first_step>", "details": {{}} }},
    {{ "desire": "<second_step>", "details": {{}} }}
  ]
}}

Your response must be ONLY the JSON object.
"""

MAPPER_DESCRIBE_CHANGE_PROMPT_TEMPLATE = """
You are a visual analysis AI. A key press ('{key_pressed}') was just executed.
Analyze the `before_image` and `after_image` to determine what happened.
Describe the change factually and concisely.
Examples: "The inventory screen appeared.", "The character jumped.", "The selected menu item moved down."
If no meaningful change occurred, state that.

Your response MUST be a single JSON object.
**Format:**
{{
  "change_description": "<Your factual description of the change.>"
}}
"""

MAPPER_NAME_ACTION_PROMPT_TEMPLATE = """
You are a game analysis AI. A key press ('{key_pressed}') caused a specific visual change.
Your task is to give this action a concise, universal, semantic name.
This name will be used to identify this action forever. Use snake_case.
Examples:
- Change: "The inventory screen appeared." -> Name: "toggle_inventory"
- Change: "The character jumped." -> Name: "jump"
- Change: "The selected menu item moved down." -> Name: "navigate_menu_down"

**Visual Change Description:**
{change_description}

Your response MUST be a single JSON object.
**Format:**
{{
  "action_name": "<your_semantic_action_name>"
}}
"""

PLAYER_EXECUTE_PROMPT_TEMPLATE = """
You are a tactical game-playing AI. Your goal is to execute a single high-level desire by selecting the best single low-level action from the KNOWN controls.
Analyze the `game_screenshot` and use the `control_schema` to achieve the `current_desire`.

**Current Desire:**
{desire_json}

**Known Controls (Control Schema):**
{control_schema_json}

Your task is to choose the single best input from the KNOWN controls to progress the desire.
Your response MUST be a single JSON object conforming to the LowLevelAction schema.
**Format:**
{{
  "type": "<'key_press' or 'key_sequence'>",
  "details": {{...}}
}}
"""

PLAYER_VERIFY_PROMPT_TEMPLATE = """
You are a visual verification AI. An action was just performed to achieve a specific desire.
Analyze the `game_screenshot` to determine if the desire was successfully achieved.

**Original Desire:**
{desire_json}

**Action Performed:**
{action_performed_json}

Your response MUST be a single JSON object.
**Format:**
{{
  "success": <true_or_false>,
  "reason": "<Briefly explain why it succeeded or failed based on the visual evidence.>"
}}
"""

# --- HELPER FUNCTIONS ---
def panic_switch_listener(stop_event: mp.Event):
    """Listens for Ctrl+Alt+Q to terminate the agent."""
    hotkey = {Key.ctrl, Key.alt, KeyCode.from_char('q')}
    current_keys = set()
    logging.info("Panic switch active. Press Ctrl+Alt+Q to terminate.")
    def on_press(key):
        if key in hotkey:
            current_keys.add(key)
            if all(k in current_keys for k in hotkey):
                logging.warning("PANIC SWITCH HOTKEY DETECTED! Shutting down.")
                stop_event.set()
                listener.stop()
    def on_release(key):
        try: current_keys.remove(key)
        except KeyError: pass
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

async def execute_action(action: LowLevelAction, controller: InputController):
    """Executes a game action using the InputController."""
    if not action:
        logging.warning("execute_action received an empty action.")
        return
    await asyncio.to_thread(controller.execute_action, action.model_dump())

def get_latest_frame(image_queue: mp.Queue, images_deque: Deque[Image.Image]) -> Image.Image | None:
    """Drains the queue and returns the most recent frame."""
    while not image_queue.empty():
        try:
            data_packet = image_queue.get_nowait()
            images_deque.append(Image.frombytes(data_packet['mode'], data_packet['size'], data_packet['image_bytes']))
        except (KeyError, EOFError):
            continue # Ignore corrupted data packets
    return images_deque[-1] if images_deque else None

# --- SPECIALIZED AGENT CLASSES (REVISED) ---

class PlannerAgent:
    """The Strategist: Creates multi-step plans based on memory and context."""
    def __init__(self, llm: JohnVLLM):
        self.llm = llm
        self.gen_config = {"max_tokens": 512, "temperature": 0.2}

    async def generate_plan(self, latest_frame: Image.Image, world_model: dict) -> Plan | None:
        logging.info("--- PLANNER: Generating new plan... ---")
        # Sanitize world model for the prompt
        prompt_model = {
            "current_context": world_model.get("current_context"),
            "control_schema": world_model.get("control_schema"),
            "action_history": list(world_model.get("action_history", [])) # Convert deque to list for JSON
        }
        prompt = PLANNER_PROMPT_TEMPLATE.format(world_model_json=json.dumps(prompt_model, indent=2))
        assistant_prompt = '{\n  "thought": "'
        
        output = self.llm.dialogue_generator(
            prompt=prompt, assistant_prompt=assistant_prompt, images=[latest_frame],
            add_generation_prompt=False, continue_final_message=True, generation_config=self.gen_config
        )
        try:
            parsed_response = Plan.model_validate_json(assistant_prompt + output)
            logging.info(f"Planner created new plan: {parsed_response.thought}")
            return parsed_response
        except (ValidationError, json.JSONDecodeError) as e:
            logging.error(f"Planner failed to generate a valid plan: {e}\nRaw output: {output}")
            return None

class MapperAgent:
    """The Cartographer: Discovers controls on-demand when the agent is stuck."""
    def __init__(self, llm: JohnVLLM, controller: InputController):
        self.llm = llm
        self.controller = controller
        self.gen_config = {"max_tokens": 128, "temperature": 0.1}
        self.candidate_keys = [
            'e', 'f', 'i', 'm', 'tab', 'escape', 'space', 'enter', 'c', 'j', 'q', 'r', 'x', 'z',
            'w', 'a', 's', 'd', 'up', 'down', 'left', 'right', 'shift', 'ctrl'
        ]

    async def _test_key(self, key: str, image_queue: mp.Queue, images_deque: Deque[Image.Image]) -> tuple[str, dict] | None:
        """Tests a single key, asks the VLM what happened, names the action, and returns the mapping."""
        logging.info(f"Mapper: Testing key '{key}'...")
        key = key[0]
        before_frame = get_latest_frame(image_queue, images_deque)
        if not before_frame:
            logging.warning(f"Mapper: Could not get 'before' frame for key '{key}'.")
            return None

        action = LowLevelAction(type="key_press", details={"key": key})
        await execute_action(action, self.controller)
        await asyncio.sleep(1.0)

        after_frame = get_latest_frame(image_queue, images_deque)
        if not after_frame:
            logging.warning(f"Mapper: Could not get 'after' frame for key '{key}'.")
            return None

        # Step 1: Describe the change
        prompt = MAPPER_DESCRIBE_CHANGE_PROMPT_TEMPLATE.format(key_pressed=key)
        assistant_prompt = '{\n  "change_description": "'
        output = self.llm.dialogue_generator(
            prompt=prompt, assistant_prompt=assistant_prompt, images=[before_frame, after_frame],
            add_generation_prompt=False, continue_final_message=True, generation_config=self.gen_config
        )
        try:
            change_result = json.loads(assistant_prompt + output)
            change_desc = change_result.get("change_description", "no change")
            
            if "no change" in change_desc.lower() or "meaningful change" not in change_desc.lower():
                logging.info(f"Mapper: No meaningful change detected for key '{key}'.")
                return None

            logging.info(f"Mapper: Detected change for key '{key}'. Description: {change_desc}")
            
            # Step 2: Name the detected action
            name_prompt = MAPPER_NAME_ACTION_PROMPT_TEMPLATE.format(
                key_pressed=key, change_description=change_desc
            )
            name_assistant_prompt = '{\n  "action_name": "'
            name_output = self.llm.dialogue_generator(
                prompt=name_prompt, assistant_prompt=name_assistant_prompt, images=None,
                add_generation_prompt=False, continue_final_message=True, generation_config=self.gen_config
            )
            name_result = json.loads(name_assistant_prompt + name_output)
            action_name = name_result["action_name"]
            
            # Press escape to try to reset state (e.g., close an opened menu)
            await execute_action(LowLevelAction(type="key_press", details={"key": "escape"}), self.controller)
            await asyncio.sleep(1.0)
            
            return action_name, action.model_dump()

        except (ValidationError, json.JSONDecodeError) as e:
            logging.error(f"Mapper failed to parse response for key '{key}': {e}")
            return None

    async def discover_control_for_desire(self, desire_to_map: str, image_queue: mp.Queue, images_deque: Deque[Image.Image], world_model: dict) -> tuple[str, dict] | None:
        """Performs a targeted search to find a control for a specific desire."""
        logging.info(f"--- MAPPER: Starting targeted search for desire: '{desire_to_map}' ---")
        
        # Suggest a few likely keys to try first to be more efficient
        suggest_prompt = f"From this list of candidates, suggest the 3 most likely keys to achieve '{desire_to_map}': {self.candidate_keys}. Output ONLY a JSON list of strings. Example: [\"key1\", \"key2\"]"
        assistant_prompt = '['
        output = self.llm.dialogue_generator(
            prompt=suggest_prompt, assistant_prompt=assistant_prompt, images=[get_latest_frame(image_queue, images_deque)],
            add_generation_prompt=False, continue_final_message=True, generation_config={"max_tokens": 50, "temperature": 0.0}
        )
        try:
            suggested_keys = json.loads(assistant_prompt + output)
        except json.JSONDecodeError:
            suggested_keys = self.candidate_keys[:3] # Fallback
        
        logging.info(f"Mapper will test these keys first: {suggested_keys}")

        for key in suggested_keys:
            if key in [details.get('details', {}).get('key') for details in world_model['control_schema'].values()]:
                continue # Skip already known keys

            mapping = await self._test_key(key, image_queue, images_deque)
            if mapping:
                action_name, input_details = mapping
                world_model['control_schema'][action_name] = input_details # Directly update the world model
                logging.info(f"Mapper discovered and added new control: '{action_name}' -> {key}")

                # Ask VLM if this new control is relevant to the original desire
                validate_prompt = f"Does the action '{action_name}' help achieve the desire '{desire_to_map}'? Respond with only 'yes' or 'no'."
                validate_output = self.llm.dialogue_generator(
                    prompt=validate_prompt, assistant_prompt="", images=[get_latest_frame(image_queue, images_deque)],
                    add_generation_prompt=False, continue_final_message=False, generation_config={"max_tokens": 5, "temperature": 0.0}
                )
                if "yes" in validate_output.lower():
                    logging.info(f"Found relevant control '{action_name}' for desire '{desire_to_map}'.")
                    return action_name, input_details

        logging.warning(f"Mapper: Targeted search for '{desire_to_map}' failed to find a relevant new control after testing likely keys.")
        return None

class PlayerAgent:
    """The Hands: Executes a single desire from a plan."""
    def __init__(self, llm: JohnVLLM, controller: InputController):
        self.llm = llm
        self.controller = controller
        self.gen_config = {"max_tokens": 256, "temperature": 0.1}

    async def execute_desire(self, desire: Desire, world_model: dict, image_queue: mp.Queue, images_deque: Deque[Image.Image]) -> ActionOutcome:
        logging.info(f"--- PLAYER: Executing desire: {desire.desire} ---")
        
        latest_frame = get_latest_frame(image_queue, images_deque)
        if not latest_frame:
            return ActionOutcome(success=False, reason="Could not get game frame.")

        # Check if a known control seems to match the desire to avoid unnecessary VLM calls
        if not any(desire.desire.lower().replace("_", "") in key.lower().replace("_", "") for key in world_model['control_schema'].keys()):
            logging.warning(f"Player: No known control seems to match desire '{desire.desire}'. Requesting discovery.")
            return ActionOutcome(success=False, reason="Control for desire is not in the known schema.", missing_control_for_desire=desire.desire)

        prompt = PLAYER_EXECUTE_PROMPT_TEMPLATE.format(
            desire_json=desire.model_dump_json(indent=2),
            control_schema_json=json.dumps(world_model['control_schema'], indent=2)
        )
        assistant_prompt = '{\n  "type": "'
        output = self.llm.dialogue_generator(
            prompt=prompt, assistant_prompt=assistant_prompt, images=[latest_frame],
            add_generation_prompt=False, continue_final_message=True, generation_config=self.gen_config
        )
        try:
            action_to_perform = LowLevelAction.model_validate_json(assistant_prompt + output)
            logging.info(f"Player: Performing action: {action_to_perform.model_dump()}")
        except (ValidationError, json.JSONDecodeError) as e:
            logging.error(f"Player failed to generate a valid action: {e}")
            return ActionOutcome(success=False, reason=f"VLM failed to choose a valid action. Error: {e}")

        await execute_action(action_to_perform, self.controller)
        await asyncio.sleep(1.5)

        after_frame = get_latest_frame(image_queue, images_deque)
        if not after_frame:
            return ActionOutcome(success=False, reason="Could not get frame for verification.")
        
        verify_prompt = PLAYER_VERIFY_PROMPT_TEMPLATE.format(
            desire_json=desire.model_dump_json(indent=2),
            action_performed_json=action_to_perform.model_dump_json(indent=2)
        )
        verify_assistant_prompt = '{\n  "success": '
        verify_output = self.llm.dialogue_generator(
            prompt=verify_prompt, assistant_prompt=verify_assistant_prompt, images=[after_frame],
            add_generation_prompt=False, continue_final_message=True, generation_config=self.gen_config
        )
        try:
            outcome = ActionOutcome.model_validate_json(verify_assistant_prompt + verify_output)
            if outcome.success:
                logging.info(f"Player: Successfully executed desire '{desire.desire}'. Reason: {outcome.reason}")
            else:
                logging.warning(f"Player: Action failed to achieve desire. Reason: {outcome.reason}.")
            return outcome
        except (ValidationError, json.JSONDecodeError) as e:
            logging.error(f"Player failed to parse verification response: {e}")
            return ActionOutcome(success=False, reason=f"Failed to parse verification from VLM. Error: {e}")

# --- MAIN AGENT LOGIC (REVISED HIGH-LEVEL CONTROLLER) ---

async def run_gaming_agent():
    """Main async function to run the modular, state-driven gaming agent."""
    controller = InputController()
    images_deque = deque(maxlen=20)
    image_data_queue = mp.Queue()
    command_queue = mp.Queue()
    stop_event = mp.Event()

    listener_thread = threading.Thread(target=panic_switch_listener, args=(stop_event,), daemon=True)
    listener_thread.start()

    capture_worker = GameCaptureWorker(
        image_data_queue=image_data_queue, command_queue=command_queue, stop_event=stop_event,
        interval_sec=0.2, source_type=1, target_size=(512, 384)
    )

    try:
        capture_worker.start()
        logging.info(f"Started capture worker with PID: {capture_worker.pid}")

        model_init_kwargs = {"gpu_memory_utilization": 0.97, "max_model_len": 16384, "trust_remote_code": True}
        model_config = BaseModelConfig(model_init_kwargs=model_init_kwargs)
        llm = JohnVLLM(model_config).load_model(model_config)
        logging.info("Model loaded. Initializing agents.")

        planner = PlannerAgent(llm)
        mapper = MapperAgent(llm, controller)
        player = PlayerAgent(llm, controller)

        # The World Model: The agent's central memory and state
        world_model = {
            "current_context": "initializing",
            "control_schema": {},
            "task_queue": deque(),
            "action_history": deque(maxlen=10), # Stores (Desire, ActionOutcome) tuples
        }
        
        agent_state = AgentState.DETERMINING_CONTEXT
        desire_to_discover: str | None = None

        while not stop_event.is_set():
            latest_frame = get_latest_frame(image_data_queue, images_deque)
            if not latest_frame:
                logging.info("Waiting for the first game frame...")
                await asyncio.sleep(1)
                continue

            match agent_state:
                case AgentState.DETERMINING_CONTEXT:
                    logging.info("State: DETERMINING_CONTEXT - Analyzing screen...")
                    prompt = CONTEXT_PROMPT_TEMPLATE
                    assistant_prompt = '{\n  "context": "'
                    output = llm.dialogue_generator(
                        prompt=prompt, assistant_prompt=assistant_prompt, images=[latest_frame],
                        add_generation_prompt=False, continue_final_message=True, generation_config={"max_tokens": 50}
                    )
                    try:
                        context_result = json.loads(assistant_prompt + output)
                        world_model["current_context"] = context_result["context"]
                        logging.info(f"New context: {world_model['current_context']}")
                    except (json.JSONDecodeError, KeyError) as e:
                        logging.error(f"Could not determine context: {e}")
                        world_model["current_context"] = "unknown"
                    
                    agent_state = AgentState.PLANNING

                case AgentState.PLANNING:
                    if not world_model["task_queue"]:
                        plan = await planner.generate_plan(latest_frame, world_model)
                        if plan and plan.steps:
                            world_model["task_queue"].extend(plan.steps)
                            logging.info(f"New plan received with {len(plan.steps)} steps. Transitioning to EXECUTING.")
                            agent_state = AgentState.EXECUTING
                        else:
                            logging.warning("Planner failed to produce a valid plan. Retrying in 10 seconds.")
                            await asyncio.sleep(10)
                            agent_state = AgentState.DETERMINING_CONTEXT # Re-assess before re-planning
                    else:
                        # Task queue already has items, proceed to execution
                        agent_state = AgentState.EXECUTING

                case AgentState.EXECUTING:
                    if not world_model["task_queue"]:
                        logging.info("Plan complete. Returning to DETERMINING_CONTEXT.")
                        agent_state = AgentState.DETERMINING_CONTEXT
                        continue

                    current_desire = world_model["task_queue"].popleft()
                    outcome = await player.execute_desire(current_desire, world_model, image_data_queue, images_deque)
                    
                    world_model["action_history"].append({"desire": current_desire.model_dump(), "outcome": outcome.model_dump()})

                    if not outcome.success:
                        world_model["task_queue"].clear() # The current plan has failed.
                        desire_to_discover = outcome.missing_control_for_desire
                        agent_state = AgentState.HANDLE_FAILURE
                    
                    await asyncio.sleep(2) # Pause between actions

                case AgentState.HANDLE_FAILURE:
                    logging.warning("State: HANDLE_FAILURE - A plan step failed.")
                    if desire_to_discover:
                        logging.info(f"Failure due to missing control for '{desire_to_discover}'. Transitioning to DISCOVERING_CONTROL.")
                        agent_state = AgentState.DISCOVERING_CONTROL
                    else:
                        logging.info("Failure was not due to a missing control. Re-planning.")
                        agent_state = AgentState.DETERMINING_CONTEXT

                case AgentState.DISCOVERING_CONTROL:
                    if not desire_to_discover:
                        logging.error("Entered DISCOVERING state without a target desire. Returning to context check.")
                        agent_state = AgentState.DETERMINING_CONTEXT
                        continue
                    
                    new_mapping = await mapper.discover_control_for_desire(desire_to_discover, image_data_queue, images_deque, world_model)
                    
                    if new_mapping:
                        logging.info(f"Successfully found a relevant control for '{desire_to_discover}'.")
                    else:
                        logging.warning(f"Failed to find a control for '{desire_to_discover}'. The agent may need to try a different strategy.")
                    
                    desire_to_discover = None
                    logging.info("State: DISCOVERING_CONTROL -> DETERMINING_CONTEXT to form a new plan with updated controls.")
                    agent_state = AgentState.DETERMINING_CONTEXT

    except Exception as e:
        logging.error(f"An unhandled error occurred in the main loop: {e}", exc_info=True)
    finally:
        logging.info("Cleaning up...")
        if 'capture_worker' in locals() and capture_worker.is_alive():
            stop_event.set()
            if controller: controller.close()
            capture_worker.join(timeout=5)
            if capture_worker.is_alive():
                logging.warning("Worker did not terminate gracefully, terminating.")
                capture_worker.terminate()
        logging.info("Cleanup complete.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    asyncio.run(run_gaming_agent())