import os
import signal
import time
import uuid
from typing import Optional, Dict, Any

import dbus
import numpy as np
from PIL import Image
from dbus.mainloop.glib import DBusGMainLoop

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

from queue import Empty, Full
import multiprocessing as mp
from multiprocessing import shared_memory
# Initialize GStreamer and DBus loop
# Note: These should be initialized within the process that uses them.
# We will call them again inside the worker.
Gst.init(None)
DBusGMainLoop(set_as_default=True)


class ScreenCastCapture:
    """
    Handles the low-level screen casting via DBus and GStreamer.
    (This class is left unchanged as its internal logic is correct)
    """
    def __init__(self, source_type: int = 1, cursor_mode: int = 2, target_size: Optional[tuple[int, int]] = (1000,1000)):
        """
        source_type: 1 (monitor/full-screen), 2 (window)
        cursor_mode: 1 (hidden), 2 (embedded), 4 (metadata)
        """
        self.bus = dbus.SessionBus()
        self.portal = self.bus.get_object('org.freedesktop.portal.Desktop', '/org/freedesktop/portal/desktop')
        self.portal_interface = dbus.Interface(self.portal, 'org.freedesktop.portal.ScreenCast')
        self.request_interface = 'org.freedesktop.portal.Request'
        self.session = None
        self.pipeline = None
        self.node_id = None
        self.source_type = source_type
        self.cursor_mode = cursor_mode
        self.loop = GLib.MainLoop()
        self.response_handlers = {}
        self.request_path_map = {} # To map request paths back to tokens
        self.target_size = target_size
        signal.signal(signal.SIGINT, self.stop_capture)

    def _generate_token(self) -> str:
        return str(uuid.uuid4()).replace('-', '_')

    def _request_path(self, token: str) -> str:
        unique_name = self.bus.get_unique_name()[1:].replace('.', '_')
        path = f'/org/freedesktop/portal/desktop/request/{unique_name}/{token}'
        self.request_path_map[path] = token # Map path to token
        return path

    def _add_response_handler(self, token: str, callback):
        path = self._request_path(token)
        self.response_handlers[token] = callback
        self.bus.add_signal_receiver(
            self._handle_response,
            signal_name='Response',
            dbus_interface=self.request_interface,
            path=path,
            path_keyword='path' # Ask dbus-python to pass the object path
        )

    def _handle_response(self, response_code: int, results: dict, path=None):
        if path and path in self.request_path_map:
            token = self.request_path_map[path]
            if token in self.response_handlers:
                callback = self.response_handlers[token]
                if response_code != 0:
                    print(f"Portal request failed for token {token}: code {response_code}, results {results}")
                    self.loop.quit()
                    return

                callback(results)
                # Clean up
                del self.response_handlers[token]
                del self.request_path_map[path]
        else:
            print(f"Warning: Received a response for an unknown request path: {path}")


    def create_session(self, callback):
        token = self._generate_token()
        self._add_response_handler(token, lambda res: callback(res['session_handle']))
        options = {'handle_token': token, 'session_handle_token': self._generate_token()}
        self.portal_interface.CreateSession(options)

    def select_sources(self, session_handle: str, callback):
        token = self._generate_token()
        self._add_response_handler(token, lambda _: callback())
        options = {
            'handle_token': token,
            'types': dbus.UInt32(self.source_type),
            'cursor_mode': dbus.UInt32(self.cursor_mode),
            'multiple': False,
        }
        self.portal_interface.SelectSources(session_handle, options)

    def start_session(self, session_handle: str, callback):
        token = self._generate_token()
        self._add_response_handler(token, lambda res: callback(res['streams']))
        options = {'handle_token': token}
        self.portal_interface.Start(session_handle, '', options)

    # def setup_pipeline(self, node_id: int, fd: int):
    #     print(f"Setting up pipeline with PipeWire node ID: {node_id} and FD: {fd}")
    #     pipeline_str = (
    #         f'pipewiresrc fd={fd} path={node_id} do-timestamp=true ! '
    #         'videoconvert ! video/x-raw,format=RGB ! '
    #         'appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true'
    #     )
    #     self.pipeline = Gst.parse_launch(pipeline_str)
    #     self.pipeline.set_state(Gst.State.PLAYING)
    #     print("Pipeline state set to PLAYING.")
    #     GLib.timeout_add(500, self.loop.quit)

    def setup_pipeline(self, node_id: int, fd: int):
        print(f"Setting up pipeline with PipeWire node ID: {node_id} and FD: {fd}")
        
        # --- NEW PIPELINE ---
        # It now includes videoscale and a sized video/x-raw caps filter.
        # This offloads all resizing work to the highly optimized GStreamer engine.
        if self.target_size:
            width, height = self.target_size
            resize_str = f'videoscale ! video/x-raw,width={width},height={height},format=RGB ! '
        else:
            resize_str = 'video/x-raw,format=RGB ! '

        pipeline_str = (
            f'pipewiresrc fd={fd} path={node_id} do-timestamp=false ! '
            f'videoconvert ! {resize_str}'
            'appsink name=sink emit-signals=true sync=false max-buffers=60 drop=false'
        )
        # --- END NEW PIPELINE ---

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.pipeline.set_state(Gst.State.PLAYING)
        print("Pipeline state set to PLAYING.")
        GLib.timeout_add(500, self.loop.quit)

    def capture_frame(self) -> Optional[np.ndarray]:
        if not self.pipeline or self.pipeline.get_state(0)[1] != Gst.State.PLAYING:
            return None
        sink = self.pipeline.get_by_name('sink')
        sample = sink.emit('pull-sample')
        if not sample:
            return None
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        height, width = caps.get_value('height'), caps.get_value('width')
        data = buf.extract_dup(0, buf.get_size())
        return np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))

    def start_capture(self):
        def on_create(session_handle):
            self.session = session_handle
            self.select_sources(session_handle, lambda: self.start_session(session_handle, on_start))

        def on_start(streams):
            if not streams:
                print("No streams available.")
                self.loop.quit()
                return
            stream_props = streams[0]
            self.node_id = stream_props[0]
            fd_obj = self.portal_interface.OpenPipeWireRemote(self.session, {}, get_handles=True)
            fd = fd_obj.take()
            self.setup_pipeline(self.node_id, fd)

        self.create_session(on_create)
        print("Starting GLib MainLoop for setup. A portal dialog should appear.")
        print("Please approve the screen sharing request.")
        self.loop.run()

    def stop_capture(self, *args):
        print("Stopping capture...")
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.session:
            try:
                session_iface = dbus.Interface(self.bus.get_object('org.freedesktop.portal.Desktop', self.session), 'org.freedesktop.portal.Session')
                session_iface.Close()
            except dbus.exceptions.DBusException as e:
                print(f"Could not close session (it may already be closed): {e}")
        if self.loop.is_running():
            self.loop.quit()

class GameCaptureWorker(mp.Process):
    """
    An independent worker process for capturing the screen.

    This worker uses shared memory for zero-copy frame transfer, making it
    extremely fast. It writes the frame data to a shared buffer and sends
    a small notification on a separate queue.
    """
    def __init__(self,
                 shm_name: str,
                 frame_shape: tuple[int, int, int],
                 frame_dtype: np.dtype,
                 notification_queue: mp.Queue,
                 command_queue: mp.Queue,
                 stop_event: mp.Event,
                 source_type: int = 1,
                 interval_sec: float = 0.016):
        super().__init__()
        # --- New Shared Memory parameters ---
        self.shm_name = shm_name
        self.frame_shape = frame_shape
        self.frame_dtype = frame_dtype
        self.target_size = (frame_shape[1], frame_shape[0]) # (width, height) for PIL

        # --- Queues and Events ---
        self.notification_queue = notification_queue
        self.command_queue = command_queue
        self.stop_event = stop_event
        
        # --- Other params ---
        self.source_type = source_type
        self.interval_sec = interval_sec
        self.capturer = None

        # These will be initialized in the run() method
        self.shm = None
        self.shm_array = None

    def run(self):
        """The main loop of the worker process."""
        Gst.init(None)
        DBusGMainLoop(set_as_default=True)
        
        try:
            print(f"[CaptureWorker-{os.getpid()}] Starting...")
            
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.shm_array = np.ndarray(self.frame_shape, dtype=self.frame_dtype, buffer=self.shm.buf)
            
            # --- Pass target_size to the capturer ---
            self.capturer = ScreenCastCapture(
                source_type=self.source_type,
                target_size=self.target_size
            )
            self.capturer.start_capture()

            while not self.stop_event.is_set():
                # start_time = time.perf_counter()
                self._handle_commands()

                frame = self.capturer.capture_frame()
                if frame is not None:
                    # --- THE ULTIMATE LEAN PRODUCER ---
                    # 1. The frame is ALREADY the right size from GStreamer.
                    # 2. Perform a single, high-speed memory copy. NO PIL. NO RESIZING.
                    self.shm_array[:] = frame
                    
                    # 3. Send a tiny, fast notification.
                    try:
                        self.notification_queue.put_nowait(time.time())
                    except Full:
                        pass

                # elapsed = time.perf_counter() - start_time
                # sleep_time = self.interval_sec - elapsed
                # if sleep_time > 0:
                    # time.sleep(sleep_time)

        except Exception as e:
            print(f"[CaptureWorker-{os.getpid()}] An error occurred: {e}")
        finally:
            print(f"[CaptureWorker-{os.getpid()}] Stopping...")
            if self.capturer:
                self.capturer.stop_capture()
            
            # --- CRITICAL CLEANUP ---
            # Each process must close its own view of the shared memory.
            if self.shm:
                self.shm.close()

    def _handle_commands(self):
        """Check for and execute commands from the main process."""
        try:
            command = self.command_queue.get_nowait()
            if command['action'] == 'set_interval':
                new_interval = command['value']
                print(f"[CaptureWorker-{os.getpid()}] Updating interval to {new_interval}s")
                self.interval_sec = new_interval
        except Empty:
            pass


class CaptureManager:
    """
    Manages the lifecycle of the GameCaptureWorker process using Shared Memory
    for high-speed, zero-copy frame transfer.
    """
    def __init__(self,
                 source_type: int = 1,
                 fps: int = 60,
                 target_size: Optional[tuple[int, int]] = (1280, 720)):
        
        self.target_size = target_size
        self._frame_shape = (target_size[1], target_size[0], 3)
        self._frame_dtype = np.uint8
        frame_bytes = np.prod(self._frame_shape) * np.dtype(self._frame_dtype).itemsize

        # 1. Create a shared memory block
        try:
            self.shm = shared_memory.SharedMemory(create=True, size=frame_bytes)
        except FileExistsError:
            # Clean up a potentially orphaned block from a previous crash
            # NOTE: This is a simplistic cleanup. A more robust solution might use a unique name.
            shared_memory.SharedMemory(name='psm_...').unlink() # The name is random, this is tricky
            self.shm = shared_memory.SharedMemory(create=True, size=frame_bytes)

        # 2. We use a standard queue just for tiny notification messages
        self._notification_queue = mp.Queue()
        self._command_queue = mp.Queue()
        self._stop_event = mp.Event()
        
        self._worker = GameCaptureWorker(
            # Pass the name of the shm block, not the object itself
            shm_name=self.shm.name,
            frame_shape=self._frame_shape,
            frame_dtype=self._frame_dtype,
            notification_queue=self._notification_queue,
            command_queue=self._command_queue,
            stop_event=self._stop_event,
            source_type=source_type,
            interval_sec=1.0 / fps
            # Note: We remove target_size from the worker; it no longer handles resizing
        )
        self._last_frame_data: Optional[Dict[str, Any]] = None

    # We need to expose the resources for the consumer
    @property
    def notification_queue(self):
        return self._notification_queue

    def start(self):
        """Starts the capture worker process."""
        print("[CaptureManager] Starting worker process...")
        self._stop_event.clear()
        self._worker.start()
        print("[CaptureManager] Worker process started.")

    def stop(self):
        """Stops the worker and crucially, cleans up the shared memory."""
        print("[CaptureManager] Stopping worker process...")
        self._stop_event.set()
        self._worker.join(timeout=5)
        if self._worker.is_alive():
            print("[CaptureManager] Worker did not terminate gracefully, terminating.")
            self._worker.terminate()
        print("[CaptureManager] Worker process stopped.")

        # --- CRITICAL CLEANUP ---
        print("[CaptureManager] Cleaning up shared memory block.")
        self.shm.close()
        self.shm.unlink() # This removes the block from the system

    # The old get_latest_frame is no longer relevant in this high-speed design.
    # The consumer will interact directly with the queue and memory.

# --- Example Usage ---
if __name__ == "__main__":
    # It's crucial to put multiprocessing code under this block
    # to prevent issues on some platforms.
    
    capture_manager = CaptureManager(fps=60, target_size=(1000, 1000))
    
    try:
        capture_manager.start()
        
        # Give the worker a moment to start up and for the user to approve the portal
        print("\nWaiting for capture to initialize (approve the dialog)...")
        time.sleep(5) 

        print("\nStarting main loop to fetch frames...")
        frames_processed = 0
        start_time = time.time()

        while frames_processed < 60: # Run for a limited number of iterations for this example
            # Simulate the main application doing some work
            # time.sleep(0.05) # ~20 FPS processing rate

            latest_frame = capture_manager.get_latest_frame()

            if latest_frame:
                frames_processed += 1
                print(f"Got a new frame! Size: {latest_frame.size}. Total processed: {frames_processed}")
                # In a real app, you would do something with the frame here,
                # like run inference, display it, etc.
                # latest_frame.save(f"frame_{i:03d}.png") # Uncomment to save frames
            else:
                # print("No new frame available yet.")
                pass

            # Example of dynamically changing FPS after 50 frames
            # if i == 50:
            #     print("\n" + "="*20)
            #     print("CHANGING CAPTURE RATE TO 10 FPS")
            #     print("="*20 + "\n")
            #     capture_manager.set_fps(10)


        end_time = time.time()
        duration = end_time - start_time
        print(f"\nProcessed {frames_processed} frames in {duration:.2f} seconds.")

    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt, shutting down.")
    finally:
        capture_manager.stop()
