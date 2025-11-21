from gaming.game_capture import CaptureManager, SystemConfig
import cv2
import numpy as np
import time

class MultiAnchorOdometry:
    def __init__(self, capture_manager):
        self.manager = capture_manager
        
        # --- The Sprite Bank ---
        # List of dicts: {'label': 'front', 'image': numpy_array}
        self.anchor_bank = [] 
        self.active_template = None   # The immediate frame-to-frame tracker
        self.player_bbox = None       # (x, y, w, h)
        
        # --- Optical Flow (Background) ---
        # self.lk_params = dict(winSize=(21, 21), maxLevel=3,
        #                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        # self.feature_params = dict(maxCorners=200, qualityLevel=0.05, minDistance=7, blockSize=7)
        
        self.lk_params = dict(winSize=(31, 31), maxLevel=4,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.feature_params = dict(maxCorners=300, qualityLevel=0.05, minDistance=20, blockSize=7)
        
        # --- State ---
        self.true_velocity = np.array([0.0, 0.0])
        self.world_position = np.array([0.0, 0.0])
        self.prev_gray = None
        
        # --- Tuning ---
        self.CONFIDENCE_THRESHOLD = 0.6
        self.game_roi_y_start = 219 + 5
        self.game_roi_y_end = (219 + 562)-6

    def _get_frame(self):
        raw_frame = self.manager.get_snapshot(duration=0.01)
        if raw_frame is None: return None
        processed_frame = self.manager.post_process_frames([raw_frame])[0]
        if len(processed_frame.shape) == 3 and processed_frame.shape[2] == 4:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGRA2BGR)
        return processed_frame
    
    

    def register_anchor(self, bbox, label="unknown"):
        """
        Call this manually OR via VLM to add a new angle to the bank.
        """
        frame = self._get_frame()
        if frame is None: return
        
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Safety Check: Ensure bbox is within frame
        h_img, w_img = frame.shape[:2]
        if x < 0 or y < 0 or x+w > w_img or y+h > h_img:
            print(f"Invalid BBox for registry: {bbox}")
            return

        template = frame[y:y+h, x:x+w].copy()
        print("THE TEAMPLETE", x,y,w,h)
        
        # Add to bank
        self.anchor_bank.append({'label': label, 'image': template})
        
        # If this is the first one, set it as active
        if self.active_template is None:
            self.active_template = template
            self.player_bbox = (x, y, w, h)
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        print(f"Registered Anchor '{label}'. Total Bank Size: {len(self.anchor_bank)}")

    def initialize_interactive(self):
        """Manually add the first anchor to start the system."""
        self.manager.start_system()
        time.sleep(2.0)
        frame = self._get_frame()
        
        print("Select initial character pose...", frame.shape)
        bbox = cv2.selectROI("Init Anchor", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Init Anchor")
        if bbox != (0,0,0,0):
            self.register_anchor(bbox, "init_pose")

    def get_game_state(self):
        frame = self._get_frame()
        if frame is None or not self.anchor_bank: 
            return np.array([0.0,0.0]), self.world_position, frame

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        px, py, pw, ph = self.player_bbox

        # --- 1. OPTICAL FLOW (Background) ---
         # --- 1. OPTICAL FLOW (Background) ---
        camera_shift = np.array([0.0, 0.0])
        mask = np.zeros_like(self.prev_gray)
        
        # ROI Masking (Strict)
        mask[self.game_roi_y_start:self.game_roi_y_end, :] = 255 
        cv2.rectangle(mask, (px-10, py-10), (px+pw+10, py+ph+10), 0, -1)
        
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)
        self.last_flow_points = p0 

        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, p0, None, **self.lk_params)
            if p1 is not None:
                good_new = p1[st==1]
                good_old = p0[st==1]
                deltas = good_old - good_new 
                
                if len(deltas) > 0:
                    # --- ROBUST FILTERING START ---
                    
                    # 1. Calculate Magnitude of every vector
                    mags = np.linalg.norm(deltas, axis=1)
                    median_mag = np.median(mags)
                    
                    # 2. Dynamic Outlier Rejection
                    # If the world is moving (median > 0.5px), ignore points that are stuck (static UI/borders)
                    if median_mag > 0.5:
                        # Keep points that are moving at least 20% of the speed of the median
                        # This kills static black bar points
                        valid_motion = mags > (median_mag * 0.2)
                        deltas = deltas[valid_motion]
                    
                    # 3. Directional Consensus (optional but good)
                    # If we still have points, calculate the final median
                    if len(deltas) > 0:
                        camera_shift = np.median(deltas, axis=0)
        # --- 2. PLAYER TRACKING (Same as before) ---
        best_score = -1
        best_loc = (0,0)
        best_template_idx = -1
        
        # Active Match
        res_act = cv2.matchTemplate(frame, self.active_template, cv2.TM_CCOEFF_NORMED)
        _, max_val_act, _, max_loc_act = cv2.minMaxLoc(res_act)
        best_score = max_val_act
        best_loc = max_loc_act
        source = "ACTIVE"

        # Anchor Match
        for i, anchor in enumerate(self.anchor_bank):
            res = cv2.matchTemplate(frame, anchor['image'], cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_loc = max_loc
                best_template_idx = i
                source = f"ANCHOR_{anchor['label']}"

        # Update Logic
        player_screen_shift = np.array([0.0, 0.0])
        if best_score > self.CONFIDENCE_THRESHOLD:
            new_x, new_y = best_loc
            if best_template_idx != -1:
                self.active_template = self.anchor_bank[best_template_idx]['image']
            elif best_score > 0.8:
                h, w = self.active_template.shape[:2]
                self.active_template = frame[new_y:new_y+h, new_x:new_x+w].copy()

            player_screen_shift = np.array([new_x - px, new_y - py])
            self.player_bbox = (new_x, new_y, self.player_bbox[2], self.player_bbox[3])
            
            cv2.rectangle(frame, (new_x, new_y), (new_x+self.player_bbox[2], new_y+self.player_bbox[3]), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (px, py), (px+pw, py+ph), (0, 0, 255), 2)

        # --- 3. INTEGRATION ---
        self.true_velocity = player_screen_shift + camera_shift
        self.world_position += self.true_velocity
        self.prev_gray = frame_gray.copy()

        # DEBUG DRAWING: Draw Optical Flow points
        if self.last_flow_points is not None:
            for i, point in enumerate(self.last_flow_points):
                a, b = point.ravel()
                # Draw green dots on features. 
                # If these are on the black bars, change game_roi_y_start!
                cv2.circle(frame, (int(a), int(b)), 3, (0, 255, 0), -1)

        return self.true_velocity, self.world_position, frame
 
    # def run_debug(self):
    #     try:
    #         while True:
    #             vel, pos, frame = self.get_game_state()
    #             if frame is None: continue
                
    #             cv2.imshow("Robust Odometry", frame)
    #             if cv2.waitKey(1) & 0xFF == ord('q'): break
    #     finally:
    #         self.manager.stop_system()
    #         cv2.destroyAllWindows()



import numpy as np
import time
from gaming.controls import InputControllerThread
input_controller = InputControllerThread()


import numpy as np
import time
import math

class NavigationController:
    def __init__(self, odometry_agent, input_controller):
        self.agent = odometry_agent
        self.input = input_controller
        self.target_world_pos = None
        self.is_navigating = False
        
        self.pixels_per_second = 200.0 # Initial speed guess
        self.move_end_timestamp = 0
        # Stop when we are this close (pixels)
        # Was 20.0 -> Now 8.0 for higher precision
        self.ARRIVAL_THRESHOLD = 9.0 
        
        # If axis difference is less than this, don't press that key.
        # Was 15 -> Now 4 (allows fine-tuning alignment)
        self.ALIGNMENT_THRESHOLD = 9.0 

    def set_target_from_screen(self, target_screen_xy):
        # We don't call get_game_state here to avoid frame skipping
        # We rely on the main loop to have updated the agent recently
        current_world_pos = self.agent.world_position
        px, py, pw, ph = self.agent.player_bbox
        
        player_screen_center = np.array([px + pw/2, py + ph/2])
        target_screen_xy = np.array(target_screen_xy)
        
        delta_vector = target_screen_xy - player_screen_center
        self.target_world_pos = current_world_pos + delta_vector
        self.is_navigating = True
        self.move_end_timestamp = 0
        print(f"NAV: Target Set. Delta: {delta_vector}")

    def update(self, current_pos):
        """
        CRITICAL CHANGE: Pass current_pos IN. Do not calculate it here.
        """
        if not self.is_navigating or self.target_world_pos is None:
            return

        # Check if we are currently in a "Move Burst"
        if time.time() < self.move_end_timestamp:
            return

        diff = self.target_world_pos - current_pos
        distance = np.linalg.norm(diff)
        
        if distance < self.ARRIVAL_THRESHOLD:
            print("NAV: Arrived.")
            self.is_navigating = False
            return

        self._execute_burst_move(diff, distance)

    def _execute_burst_move(self, diff, distance):
        dx, dy = diff
        keys_to_press = []
        
        # Check axes individually with tighter tolerance
        if abs(dx) > self.ALIGNMENT_THRESHOLD: 
            keys_to_press.append('right' if dx > 0 else 'left')
        if abs(dy) > self.ALIGNMENT_THRESHOLD: 
            keys_to_press.append('down' if dy > 0 else 'up')

        if not keys_to_press:
            self.is_navigating = False
            return

        # Calculate time
        estimated_time = distance / self.pixels_per_second
        hold_time = max(0.05, min(estimated_time, 0.8)) # Cap burst at 0.8s

        print(f"NAV: Burst {keys_to_press} for {hold_time:.2f}s")
        
        self.input.execute_action({
            "type": "key_press",
            "details": {"key": keys_to_press, "hold_time": hold_time}
        })

        # Lock navigation logic, but allow ODOMETRY to continue in main loop
        self.move_end_timestamp = time.time() + hold_time + 0.05
    def stop(self):
        self.is_navigating = False
        self.target_world_pos = None

if __name__ == "__main__":
    # (Same config as before)
    config = SystemConfig(device_index=0, target_size=(1000, 1000))
    manager = CaptureManager(config)
    agent = MultiAnchorOdometry(manager)
    navigator = NavigationController(agent, input_controller)
    input_controller.start()
    agent.initialize_interactive()
    
    time.sleep(2)

    navigator.set_target_from_screen((364, 620))
    
    # We get the "down" facing from the init pose
    time.sleep(5)
    vlm_bbox_right = (473, 428, 52, 74)
    agent.register_anchor(vlm_bbox_right, label="right_facing")
    print("GOT THE RIGHT, GOING FOR LEFT")
    time.sleep(5)
    vlm_bbox_right = (473, 428, 52, 74)
    agent.register_anchor(vlm_bbox_right, label="left_facing")
    print("GOT THE DOWN, GOING FOR UP")
    time.sleep(5)
    vlm_bbox_right = (473, 428, 52, 74)
    agent.register_anchor(vlm_bbox_right, label="up_facing")
    print("STARTING SYSTEM FOR MOVMENT")

    # agent.run_debug()
    time.sleep(5)
    # try:
    #     while True:
    #         # A. Update Odometry (Eyes)
    #         # This updates agent.world_position internally
    #         vel, pos, frame = agent.get_game_state()
            
    #         if frame is None: continue
            
    #         # B. Update Navigation (Hands)
    #         # This checks agent.world_position vs Target and presses keys
    #         navigator.update()
            
    #         # C. Debug Visualization
    #         if navigator.target_world_pos is not None:
    #             # Draw a line from Player to Target (Visualized in Screen Space)
    #             # We need to reverse the math: Screen = World - Player_World + Player_Screen
    #             px, py, pw, ph = agent.player_bbox
    #             player_screen_center = np.array([px+pw/2, py+ph/2])
                
    #             diff = navigator.target_world_pos - pos
    #             target_screen_draw = player_screen_center + diff
                
    #             cv2.line(frame, 
    #                      (int(player_screen_center[0]), int(player_screen_center[1])),
    #                      (int(target_screen_draw[0]), int(target_screen_draw[1])),
    #                      (255, 0, 255), 2)
                         
    #             cv2.circle(frame, (int(target_screen_draw[0]), int(target_screen_draw[1])), 
    #                        10, (255, 0, 255), -1)

    #         cv2.imshow("AI Vision", frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    # finally:
    #     navigator.stop_moving()
    #     manager.stop_system()
    #     cv2.destroyAllWindows()

    try:
        while True:
            # A. Update Eyes (Must run every frame for tracking)
            vel, pos, frame = agent.get_game_state()
            
            if frame is None: continue
            
            # B. Update Brain (Runs logic only when needed)
            navigator.update(pos)
            
            # 3. VISUALIZATION
            if navigator.target_world_pos is not None:
                px, py, pw, ph = agent.player_bbox
                player_center = np.array([px+pw/2, py+ph/2])
                
                # Math: Target_Screen = Player_Screen + (Target_World - Player_World)
                diff = navigator.target_world_pos - pos
                target_draw = player_center + diff
                
                cv2.line(frame, 
                         (int(player_center[0]), int(player_center[1])),
                         (int(target_draw[0]), int(target_draw[1])),
                         (255, 0, 255), 2)
                cv2.circle(frame, (int(target_draw[0]), int(target_draw[1])), 
                           8, (0, 255, 0), -1)

            cv2.imshow("AI Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    finally:
        manager.stop_system()
        cv2.destroyAllWindows()