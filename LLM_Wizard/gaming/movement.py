#skelly steerer -- TODO: get bbox system(?)
import math
import time

# --- 1. Core Component: A Vector2D Class ---
# This class is the foundation for all our calculations. It's purely mathematical.
class Vector2D:
    """A simple class for 2D vector math."""
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector2D(self.x * float(scalar), self.y * float(scalar))

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            return Vector2D(self.x / mag, self.y / mag)
        return Vector2D(0, 0)

    def set_magnitude(self, new_mag):
        return self.normalize() * new_mag

    def limit(self, max_mag):
        if self.magnitude() > max_mag:
            return self.set_magnitude(max_mag)
        return self

    def __repr__(self):
        # Provides a clean string representation for printing
        return f"Vector2D({self.x:.2f}, {self.y:.2f})"

class ActionTranslator:
    """Translates a 2D intention vector into a discrete keyboard action with dynamic hold time."""
    def __init__(self, press_threshold=0.3):
        """
        Initializes the translator.
        Args:
            press_threshold (float): How strong the vector's component must be to trigger a key press.
        """
        self.press_threshold = press_threshold

    def translate(self, intention_vector, directional_constraints, speeds_pps, tick_duration_cap=1.0):
        """
        Converts the vector into an action dictionary.
        Args:
            intention_vector (Vector2D): The intention vector from the MovementController.
            directional_constraints (str): e.g., 'eight_way', 'four_way'.
            max_speed_pps (float): The game's known max speed in pixels per second.
            tick_duration_cap (float): The maximum hold time for a single decision tick.
        
        Returns:
            dict or None: An action dictionary or None if no action.
        """
        keys_to_press = []
        # ... (Key selection logic remains the same as before) ...
        # Horizontal Movement
        if intention_vector.x < -self.press_threshold:
            keys_to_press.append("left")
        elif intention_vector.x > self.press_threshold:
            keys_to_press.append("right")

        # Vertical Movement
        if directional_constraints in ['eight_way', 'four_way', 'vertical_only']:
            if intention_vector.y < -self.press_threshold:
                keys_to_press.append("up")
            elif intention_vector.y > self.press_threshold:
                keys_to_press.append("down")
        
        # Handle 'four_way' constraint (no diagonals)
        if directional_constraints == 'four_way' and len(keys_to_press) > 1:
            if abs(intention_vector.x) > abs(intention_vector.y):
                keys_to_press = [k for k in keys_to_press if k in ["left", "right"]]
            else:
                keys_to_press = [k for k in keys_to_press if k in ["up", "down"]]
        
        if not keys_to_press:
            return None

        # --- NEW: Dynamic Hold Time Calculation ---
 # --- NEW: Speed Selection Logic ---
        is_horizontal = "left" in keys_to_press or "right" in keys_to_press
        is_vertical = "up" in keys_to_press or "down" in keys_to_press
        
        selected_speed = 0
        if is_horizontal and is_vertical:
            selected_speed = speeds_pps.get('diagonal', speeds_pps['horizontal']) # Fallback to horizontal
        elif is_horizontal:
            selected_speed = speeds_pps['horizontal']
        elif is_vertical:
            selected_speed = speeds_pps['vertical']

        if selected_speed <= 0: return None # Cannot move
        
        desired_distance = intention_vector.magnitude()
        hold_time = desired_distance / selected_speed
        final_hold_time = min(hold_time, tick_duration_cap)
        
        return {"key_press": sorted(keys_to_press), "hold_time": final_hold_time, "speed_used": selected_speed}

class MovementController:
    def __init__(self, config):
        self.config = config
        self.translator = ActionTranslator(press_threshold=0.3)
        # We use the fastest possible speed (diagonal) for planning our cap.
        self.planning_speed = max(self.config['speeds_pps'].values())
        # ... unsticking logic is unchanged ...
        self.stuck_check_interval = 30
        self.stuck_duration = 0
        self.stuck_threshold = 90
        self.last_position = Vector2D(0, 0)
        self.is_escaping = False
        self.escape_ticks = 0

    def decide_action(self, character_pos, target_pos, obstacles, interaction_range=15.0):
        distance_to_target = (character_pos - target_pos).magnitude()
        if distance_to_target <= interaction_range:
            return None # We have arrived.

        self._update_stuck_state(character_pos)

        if self.is_escaping:
            # Escape logic remains the same
            intention_vector = self._get_escape_vector(character_pos, target_pos)
            self.escape_ticks -= 1
            if self.escape_ticks <= 0:
                self.is_escaping = False
        else:
            # REVISED: Pass interaction_range to the calculation method
            arrive_intent = self._calculate_arrive_intention(character_pos, target_pos, interaction_range)
            avoid_intent = self._calculate_avoid_intention(character_pos, obstacles)
            intention_vector = (arrive_intent * 0.5) + (avoid_intent * 2.5)

        # print("WOWO", intention_vector, arrive_intent, avoid_intent)
        intention_vector = self._apply_directional_constraints(intention_vector)
        intention_vector = self._handle_impenetrable_collisions(character_pos, intention_vector, obstacles)
        
        # MODIFIED: Pass the entire speeds dictionary
        return self.translator.translate(
            intention_vector, 
            self.config['directional_constraints'], 
            self.config['speeds_pps'], 
            self.config.get('tick_duration_cap', 0.15)
        )

    # --- COMPLETELY REWRITTEN METHOD ---
    def _calculate_arrive_intention(self, current_pos, target_pos, interaction_range):
        """
        Calculates the intention vector based on real-world distances and game speed.
        """
        # 1. Determine the maximum distance we can possibly travel in one decision tick
        # MODIFIED: Use the pre-calculated planning_speed for the cap
        max_dist_this_tick = self.planning_speed * self.config.get('tick_duration_cap', 0.15)
        # max_dist_this_tick = self.config['max_speed_pps'] * self.config['tick_duration_cap']
        
        # 2. Determine the exact distance we need to travel to reach the interaction zone
        dist_to_target = (target_pos - current_pos).magnitude()
        dist_to_travel = dist_to_target - interaction_range
        
        # 3. The desired distance is the SMALLER of the two
        # We don't want to travel further than we need to, or further than we can.
        desired_distance = min(dist_to_travel, max_dist_this_tick)
        
        # 4. If distance is negligible, don't move
        if desired_distance < 1.0: # 1 pixel deadzone
            return Vector2D(0, 0)
            
        # 5. Create the intention vector with the correct direction and desired magnitude
        direction = (target_pos - current_pos).normalize()
        intention_vector = direction * desired_distance

        # In 'acceleration' mode, we can still apply a slowdown ramp if desired
        if self.config['movement_mode'] == 'acceleration':
            slowing_radius = 150.0 # Start slowing down when 150px away
            if dist_to_target < slowing_radius:
                # Scale the intention by how close we are
                scale = dist_to_target / slowing_radius
                return intention_vector * scale
        
        return intention_vector

    def _calculate_avoid_intention(self, current_pos, obstacles):
        total_avoid_force = Vector2D(0, 0)
        avoid_radius = 60.0 # How far the controller "sees"
        
        for obs in obstacles:
            dist = (obs['position'] - current_pos).magnitude()
            effective_radius = avoid_radius + obs['radius']
            if 0 < dist < effective_radius:
                repel_force = current_pos - obs['position']
                # Stronger repulsion for closer objects
                scale = 1 - (dist / effective_radius)
                total_avoid_force += repel_force.normalize() * scale

        return total_avoid_force

    def _apply_directional_constraints(self, vector):
        if self.config['directional_constraints'] == 'horizontal_only':
            return Vector2D(vector.x, 0)
        if self.config['directional_constraints'] == 'vertical_only':
            return Vector2D(0, vector.y)
        return vector

    def _handle_impenetrable_collisions(self, current_pos, vector, obstacles):
        # Predict position one step ahead
        prediction_step = 5.0
        predicted_pos = current_pos + vector.normalize() * prediction_step

        for obs in obstacles:
            if obs.get('impenetrable', False):
                dist_to_obstacle = (predicted_pos - obs['position']).magnitude()
                if dist_to_obstacle < obs['radius']:
                    # Collision is imminent, veto movement towards the obstacle
                    away_from_obs = (predicted_pos - obs['position']).normalize()
                    dot_product = (vector.x * away_from_obs.x) + (vector.y * away_from_obs.y)
                    if dot_product < 0: # If moving towards the obstacle
                        # Project vector to be perpendicular to the away vector
                        rejection = away_from_obs * dot_product
                        return vector - rejection # Slide along the wall
        return vector

    def _update_stuck_state(self, current_pos):
        self.stuck_check_interval -= 1
        if self.stuck_check_interval <= 0:
            distance_moved = (current_pos - self.last_position).magnitude()
            if distance_moved < 5.0: # If moved less than 5 pixels
                self.stuck_duration += 30 # Increment by the interval
            else:
                self.stuck_duration = 0 # Reset if moved enough
            
            self.last_position = current_pos
            self.stuck_check_interval = 30 # Reset interval timer

        if not self.is_escaping and self.stuck_duration >= self.stuck_threshold:
            print("--- AGENT STUCK! Activating escape maneuver. ---")
            self.is_escaping = True
            self.escape_ticks = 60 # Try to escape for 60 ticks
            self.stuck_duration = 0

    def _get_escape_vector(self, current_pos, target_pos):
        # A simple escape: move perpendicular to the target direction
        to_target = target_pos - current_pos
        # Rotate by 90 degrees (swap x/y and negate one)
        return Vector2D(-to_target.y, to_target.x).normalize()


    

def take_action(key_press):
    from gaming.controls import InputController
    input_controller = InputController()
    input_controller.execute_action({"type": "key_press", "details": {"key": key_press, "hold_time": 0.1}})
    input_controller.close()
if __name__ == '__main__':
    # --- 1. Define NEW, SIMPLIFIED Game Configuration ---
    game_config = {
        'movement_mode': 'constant_speed',
        'directional_constraints': 'eight_way',
        'speeds_pps': {
            'horizontal': 193.82,
            'vertical': 109.02,
            'diagonal': 222.38,
        },
        'tick_duration_cap': 1.0, # The absolute max time we'll ever press a key for in one tick.
    }

    # --- 2. Initialize Controller and Game State ---
    print("--- SIMULATION: Dynamically Capped Movement ---")
    controller = MovementController(game_config)
    
    character_position = Vector2D(499,436)
    target_position = Vector2D(346, 640)
    obstacles = [
        {'position': Vector2D(400, 550), 'radius': 80, 'impenetrable': True}
    ]
    interaction_range = 30.0

    print(f"START: {character_position}, TARGET: {target_position}, OBSTACLES: {len(obstacles)}")
    print("-" * 50)

    # --- 3. Run The Simulation Loop ---
    time.sleep(5.0)
    max_ticks = 500
    for i in range(max_ticks):
        # The controller makes a decision based on the current state
        action = controller.decide_action(
            character_position, 
            target_position, 
            obstacles, 
            interaction_range=interaction_range
        )
        
        # --- Mock Game Engine: Update character based on action ---
        print(f"Tick {i+1:03d} | Pos: {character_position} | Action: {action}")
        
        if action is None:
            print("\n--- ACTION IS NONE: Target reached successfully! ---")
            break
        
        # Calculate the distance to move based on the dynamic hold_time
        distance_to_move = action['speed_used'] * action['hold_time'] #game_config['max_speed_pps'] * action['hold_time']
        print("WE CHOSE", action['speed_used'])
        # Determine the direction of movement from the pressed keys
        move_vector = Vector2D(0, 0)
        if "left" in action['key_press']: move_vector.x -= 1
        if "right" in action['key_press']: move_vector.x += 1
        if "up" in action['key_press']: move_vector.y -= 1
        if "down" in action['key_press']: move_vector.y += 1
        if "left" in action['key_press']: take_action(key_press=action["key_press"])
        if "right" in action['key_press']: take_action(key_press=action["key_press"])
        if "up" in action['key_press']: take_action(key_press=action["key_press"])
        if "down" in action['key_press']: take_action(key_press=action["key_press"])
        
        # Update the character's position
        # We normalize the move_vector to get a pure direction, then scale by distance
        character_position += move_vector.normalize() * distance_to_move
    else: # This 'else' belongs to the 'for' loop, it runs if the loop finishes without a 'break'
        print("\n--- SIMULATION ENDED: Max ticks reached. ---")

    # --- 4. Final Report ---
    final_distance = (character_position - target_position).magnitude()
    print("-" * 50)
    print("--- SIMULATION COMPLETE ---")
    print(f"Final Position: {character_position}")
    print(f"Final Distance to Target: {final_distance:.2f} pixels")
    print(f"Target Interaction Range: {interaction_range:.2f} pixels")
    if int(final_distance) <= int(interaction_range):
        print("Outcome: SUCCESS")
    else:
        print("Outcome: FAILURE (Did not reach interaction range)")

