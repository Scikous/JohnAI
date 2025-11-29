import logging
import time
from typing import Dict, Any
from threading import Thread, Event
from queue import Queue

try:
    from evdev import UInput, ecodes as e
except ImportError:
    print("The 'evdev' library is not installed. Please install it with 'pip install evdev'")
    print("This module is intended for Linux systems only.")
    exit(1)

try:
    from screeninfo import get_monitors
except ImportError:
    print("The 'screeninfo' library is not installed. Please install it with 'pip install screeninfo'")
    exit(1)

class InputController:
    """
    Emulates a standard mouse and keyboard using a purely relative evdev device.
    This is the most compatible method for Wayland compositors.
    It maintains an internal state to translate absolute coordinates into
    the necessary relative movements for the mouse.
    """

    def __init__(self):
        """Initializes the virtual input device."""
        self.screen_width, self.screen_height = self._get_screen_resolution()
        
        # Internal state for tracking the virtual cursor's position.
        # We initialize it to the center of the screen.
        self.current_x = self.screen_width // 2
        self.current_y = self.screen_height // 2
        
        self.ui = None
        try:
            # Emulate a standard mouse (EV_REL) and a full keyboard (EV_KEY).
            # This combination is highly compatible with modern desktop environments.
            capabilities = {
                e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE] + list(range(e.KEY_RESERVED, e.KEY_MAX + 1)),
                e.EV_REL: [e.REL_X, e.REL_Y, e.REL_WHEEL],
            }

            self.ui = UInput(capabilities, name='AI-Agent-Controller', version=0x1)
            logging.info(f"InputController initialized. Internal cursor at: ({self.current_x}, {self.current_y})")

        except Exception as err:
            logging.error(
                f"Failed to initialize UInput device: {err}\n"
                "Please ensure you have the correct permissions for /dev/uinput.\n"
                "Did you run the one-time setup and reboot?"
            )
            raise

    def _get_screen_resolution(self) -> tuple[int, int]:
        """Detects the primary monitor's resolution."""
        try:
            monitors = get_monitors()
            primary_monitor = next((m for m in monitors if m.is_primary), monitors[0])
            return primary_monitor.width, primary_monitor.height
        except Exception:
            logging.warning("Could not auto-detect screen resolution. Falling back to 1920x1080.")
            return 1920, 1080

    def close(self):
        """Closes and destroys the virtual input device."""
        if self.ui:
            self.ui.close()
            logging.info("UInput device closed.")

    def execute_action(self, action: Dict[str, Any]):
        """Public method to dispatch an action to the appropriate handler."""
        if not self.ui:
            logging.error("Cannot execute action, UInput device is not initialized.")
            return

        action_type = action.get("type")
        details = action.get("details", {})

        handler_map = {
            "mouse_move": self._handle_mouse_move,
            "mouse_click": self._handle_mouse_click,
            "click": self._handle_mouse_click, # Alias
            "key_press": self._handle_key_press,
            "key_down": self._handle_key_down,
            "key_up": self._handle_key_up,
        }
        
        handler_method = handler_map.get(action_type)

        if handler_method:
            try:
                handler_method(details)
            except Exception as err:
                logging.error(f"Error executing action '{action_type}': {err}", exc_info=True)
        else:
            logging.warning(f"No handler found for action type: '{action_type}'")

    def _handle_mouse_move(self, details: Dict[str, Any]):
        """Translates absolute coordinates into relative movement events."""
        target_x = details.get("target_x")
        target_y = details.get("target_y")
        if target_x is None or target_y is None: return

        target_x, target_y = int(target_x), int(target_y)

        delta_x = target_x - self.current_x
        delta_y = target_y - self.current_y

        if delta_x != 0:
            self.ui.write(e.EV_REL, e.REL_X, delta_x)
        if delta_y != 0:
            self.ui.write(e.EV_REL, e.REL_Y, delta_y)
        
        if delta_x != 0 or delta_y != 0:
            self.ui.syn()

        self.current_x = target_x
        self.current_y = target_y

    def _handle_mouse_click(self, details: Dict[str, Any]):
        """Performs a click, moving to a position first if specified."""
        if "target_x" in details and "target_y" in details:
            self._handle_mouse_move(details)
            time.sleep(0.03)

        button_str = details.get("button", "left").lower()
        button_map = {"left": e.BTN_LEFT, "right": e.BTN_RIGHT, "middle": e.BTN_MIDDLE}
        button = button_map.get(button_str, e.BTN_LEFT)

        self.ui.write(e.EV_KEY, button, 1); self.ui.syn() # Press
        time.sleep(0.05)
        self.ui.write(e.EV_KEY, button, 0); self.ui.syn() # Release
        logging.info(f"Clicked {button_str} button at virtual coords ({self.current_x}, {self.current_y}).")

    def _get_keycode(self, key_str: str) -> int | None:
        """Maps a human-readable string to an evdev keycode."""
        if not key_str: return None
        
        key_str = key_str.upper()
        
        alias_map = {
            "CTRL": "LEFTCTRL", "CONTROL": "LEFTCTRL",
            "SHIFT": "LEFTSHIFT",
            "ALT": "LEFTALT",
            "WIN": "LEFTMETA", "SUPER": "LEFTMETA",
            "ENTER": "ENTER", "RETURN": "ENTER",
            "ESC": "ESC", "ESCAPE": "ESC",
            "SPACE": "SPACE",
        }
        key_str = alias_map.get(key_str, key_str)
        
        keycode = getattr(e, f'KEY_{key_str}', None)
        if not keycode:
            logging.warning(f"Unknown key: '{key_str}'")
        return keycode

    def _handle_key_press(self, details: Dict[str, Any]):
        """Handles a single, quick key press (tap)."""
        keys = [key for key in details.get("key")]
        key_codes = [self._get_keycode(key) for key in keys]
        # key_codes = [key for key in self._get_keycode(details.get("key"))]
        # key_code = self._get_keycode(details.get("key"))
        hold_time = details.get("hold_time", 0.05)
        # if not key_code: return
        for key_code in key_codes:
            self.ui.write(e.EV_KEY, key_code, 1); self.ui.syn() # Key down
        time.sleep(hold_time)
        for key_code in key_codes:
            self.ui.write(e.EV_KEY, key_code, 0); self.ui.syn() # Key up

    def _handle_key_down(self, details: Dict[str, Any]):
        """Handles pressing and holding a key down."""
        key_code = self._get_keycode(details.get("key"))
        if not key_code: return
        
        self.ui.write(e.EV_KEY, key_code, 1); self.ui.syn()

    def _handle_key_up(self, details: Dict[str, Any]):
        """Handles releasing a key."""
        key_code = self._get_keycode(details.get("key"))
        if not key_code: return
            
        self.ui.write(e.EV_KEY, key_code, 0); self.ui.syn()


# NEW: A thread-safe wrapper class for the controller
class InputControllerThread:
    """
    Runs the InputController in a separate thread to prevent blocking
    the main application. Actions are sent via a thread-safe queue.
    """
    def __init__(self):
        self.action_queue = Queue()
        self.controller = InputController()
        self._stop_event = Event()
        # A daemon thread will exit when the main program exits.
        self.worker_thread = Thread(target=self._worker_loop, daemon=True)

    def _worker_loop(self):
        """The main loop for the consumer thread."""
        logging.info("InputController worker thread started.")
        while True:
            # This will block until an item is available.
            action = self.action_queue.get()
            
            # Use a sentinel value (None) to signal the thread to stop.
            if action is None:
                # self._stop_event.set()
                break
            
            self.controller.execute_action(action)
            self.action_queue.task_done()

        self.controller.close()
        logging.info("InputController worker thread stopped.")

    def start(self):
        """Starts the worker thread."""
        self.worker_thread.start()
    def is_alive(self):
        return self.worker_thread.is_alive()

    def stop(self):
        """Stops the worker thread gracefully."""
        logging.info("Requesting controller thread to stop.")
        self._stop_event.set()
        self.action_queue.put(None) # Send the sentinel
        self.worker_thread.join()   # Wait for the thread to finish


    def execute_action(self, action: Dict[str, Any]):
        """
        Public, non-blocking method to add an action to the queue.
        """
        self.action_queue.put(action)


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - [%(filename)s] %(message)s")
    
#     controller_thread = None
#     try:
#         # Use the new threaded controller
#         controller_thread = InputControllerThread()
#         controller_thread.start()
        
#         print("Threaded controller initialized. The virtual cursor starts at screen center.")
#         print("Testing in 3 seconds... These commands will execute without blocking the main script.")
#         print("Notice how the script prints messages instantly.")
#         time.sleep(3)

#         # --- Mouse Test ---
#         print("--- Testing Mouse ---")
#         print("Queueing move to (300, 400)...")
#         controller_thread.execute_action({"type": "mouse_move", "details": {"target_x": 300, "target_y": 400}})
        
#         print("Queueing a click...")
#         # Note: we add a small delay in the main thread only if we want to see the actions visually separated.
#         # The controller thread handles its own internal delays.
#         time.sleep(1) 
#         controller_thread.execute_action({"type": "mouse_click", "details": {}})
        
#         # --- Keyboard Test ---
#         print("\n--- Testing Keyboard ---")
#         print("Queueing 'hello'...")
#         time.sleep(1)
#         # We can now create helper functions to queue complex sequences
#         for char in "hello":
#             controller_thread.execute_action({"type": "key_press", "details": {"key": char}})

#         print("Queueing ' WORLD' (with shift modifier)...")
#         controller_thread.execute_action({"type": "key_press", "details": {"key": "space"}})
#         controller_thread.execute_action({"type": "key_down", "details": {"key": "shift"}})
#         for char in "world":
#              controller_thread.execute_action({"type": "key_press", "details": {"key": char}})
#         controller_thread.execute_action({"type": "key_up", "details": {"key": "shift"}})
        
#         print("\nAll actions queued. Main script can do other work now or wait.")
#         # Wait for the queue to be empty before finishing
#         controller_thread.action_queue.join()
#         print("All queued actions have been executed.")

#     except Exception as main_err:
#         logging.error(f"An error occurred during the test: {main_err}", exc_info=True)
#     finally:
#         if controller_thread:
#             controller_thread.stop()
#         print("Script finished.")