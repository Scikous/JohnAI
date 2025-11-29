
# ## NEW SMEXY CODE
import cv2
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
from queue import Full, Empty
import threading
import psutil
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import concurrent.futures


# --- Configuration ---
@dataclass
class SystemConfig:
    device_index: int = 0
    src_width: int = 1920
    src_height: int = 1080
    target_fps: int = 30
    target_size: Tuple[int, int] = (1000, 1000)
    enable_psutil: bool = True
    worker_priority: int = -20
    worker_affinity: Optional[List[int]] = field(default_factory=lambda: [2])
    warmup_time: float = 4.0

# --- Helper for Resizing (Unchanged) ---
def process_single_frame_letterbox(frame, target_size):
    src_h, src_w = frame.shape[:2]
    dst_w, dst_h = target_size
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    x_offset = (dst_w - new_w) // 2
    y_offset = (dst_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

# --- Workers ---

class CaptureWorker(mp.Process):
    def __init__(self, config: SystemConfig, shm_name, frame_shape, frame_dtype, notification_q, stop_event, shm_lock):
        super().__init__()
        self.config = config
        self.shm_name = shm_name
        self.frame_shape = frame_shape
        self.frame_dtype = frame_dtype
        self.notification_q = notification_q
        self.stop_event = stop_event
        self.shm_lock = shm_lock # New Lock
        self.daemon = True

    def _configure_process(self):
        if not self.config.enable_psutil: return
        try:
            p = psutil.Process(os.getpid())
            p.nice(self.config.worker_priority)
            if self.config.worker_affinity:
                p.cpu_affinity(self.config.worker_affinity)
        except Exception as e:
            print(f"[CaptureWorker] PSUTIL Setup Warning: {e}")

    def run(self):
        self._configure_process()
        try:
            shm = shared_memory.SharedMemory(name=self.shm_name)
            shm_array = np.ndarray(self.frame_shape, dtype=self.frame_dtype, buffer=shm.buf)
        except FileNotFoundError:
            return

        cap = cv2.VideoCapture(self.config.device_index)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.src_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.src_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        cap.grab()

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                # CRITICAL FIX: Acquire lock before writing to shared memory
                with self.shm_lock:
                    shm_array[:] = frame
                
                try:
                    self.notification_q.put_nowait(time.perf_counter())
                except Full:
                    pass 
            else:
                time.sleep(0.01)

        cap.release()
        shm.close()

class FrameCollector(threading.Thread):
    def __init__(self, config: SystemConfig, shm_name, frame_shape, frame_dtype, notification_q, stop_event, capturing_event, shm_lock):
        super().__init__()
        self.config = config
        self.shm_name = shm_name
        self.frame_shape = frame_shape
        self.frame_dtype = frame_dtype
        self.notification_q = notification_q
        self.stop_event = stop_event
        self.capturing_event = capturing_event
        self.shm_lock = shm_lock # New Lock
        self.results = []
        self.daemon = True

    def run(self):
        shm = shared_memory.SharedMemory(name=self.shm_name)
        shm_array = np.ndarray(self.frame_shape, dtype=self.frame_dtype, buffer=shm.buf)

        while not self.stop_event.is_set():
            try:
                _ = self.notification_q.get(timeout=0.1)
                
                if self.capturing_event.is_set():
                    # CRITICAL FIX: Acquire lock before reading shared memory
                    # This waits until the Worker is done writing the full frame
                    with self.shm_lock:
                        self.results.append(shm_array.copy())
            except Empty:
                continue
            except Exception:
                break
        
        shm.close()

class CaptureManager:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.frame_shape = (config.src_height, config.src_width, 3)
        self.frame_dtype = np.uint8
        frame_bytes = int(np.prod(self.frame_shape) * np.dtype(self.frame_dtype).itemsize)
        
        try:
            self.shm = shared_memory.SharedMemory(create=True, size=frame_bytes)
        except FileExistsError:
            try:
                temp = shared_memory.SharedMemory(name='frame_shm_buffer') 
                temp.unlink()
            except: pass
            self.shm = shared_memory.SharedMemory(create=True, size=frame_bytes)

        self.notification_q = mp.Queue(maxsize=120) 
        self.stop_event = mp.Event()
        self.capturing_event = threading.Event()
        
        # NEW: The Lock preventing the race condition
        self.shm_lock = mp.Lock()

        self._worker = CaptureWorker(
            config, self.shm.name, self.frame_shape, self.frame_dtype,
            self.notification_q, self.stop_event, self.shm_lock
        )
        self._collector = FrameCollector(
            config, self.shm.name, self.frame_shape, self.frame_dtype,
            self.notification_q, self.stop_event, self.capturing_event, self.shm_lock
        )

    def start_system(self):
        print(f"System: Starting capture process...")
        self._worker.start()
        self._collector.start()
        
        if self.config.warmup_time > 0:
            print(f"System: Warming up for {self.config.warmup_time} seconds...")
            time.sleep(self.config.warmup_time)
            while not self.notification_q.empty():
                try: self.notification_q.get_nowait()
                except Empty: break
            print("System: Ready.")

    def stop_system(self):
        self.stop_event.set()
        self._worker.join(timeout=2)
        self._collector.join(timeout=2)
        if self._worker.is_alive(): self._worker.terminate()
        self.shm.close()
        try: self.shm.unlink()
        except FileNotFoundError: pass

    def start_capture(self):
        self._collector.results.clear()
        self.capturing_event.set()

    def stop_capture(self) -> List[np.ndarray]:
        self.capturing_event.clear()
        return list(self._collector.results)

    def post_process_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not frames: return []
        processed_frames = [None] * len(frames)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_single_frame_letterbox, frame, self.config.target_size): i 
                for i, frame in enumerate(frames)
            }
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                processed_frames[idx] = future.result()
        return processed_frames    

    def get_snapshot(self, duration: float = 0.1) -> Optional[np.ndarray]:
        """
        Helper to grab a single frame using the batch-oriented manager.
        It captures for a fraction of a second and returns the last frame.
        """
        frames = None
        while not frames:
            self.start_capture()
            time.sleep(duration)
            frames = self.stop_capture()
        
        if frames:
            return frames[-1] # Return the most recent frame
        return None

if __name__ == "__main__":
    
    # Configuration
    config = SystemConfig(
        device_index=0,
        src_width=2560,
        src_height=1440,
        target_fps=60,
        target_size=(1000, 1000), # VLM Preferred Size
        enable_psutil=True,       
        worker_affinity=[2],     
        warmup_time=2.0           
    )

    manager = CaptureManager(config)
    
    try:
        manager.start_system()

        for i in range(1): 
            print(f"\n--- Run {i+1} ---")
            
            manager.start_capture()
            start_time = time.perf_counter()
            
            # Capture exactly 1 second
            time.sleep(1.0)
            
            raw_frames = manager.stop_capture()
            duration = time.perf_counter() - start_time
            
            print(f"Captured {len(raw_frames)} raw frames in {duration:.4f}s")
            
            # Resize logic runs AFTER capture
            t0_proc = time.perf_counter()
            final_frames = manager.post_process_frames(raw_frames)
            proc_time = time.perf_counter() - t0_proc
            
            print(f"Processed (Letterboxed) {len(final_frames)} frames in {proc_time:.4f}s")

            # Save debug image
            if final_frames:
                save_path = "debug_frames/debug_letterbox.png"
                cv2.imwrite(save_path, final_frames[0])
                print(f"Saved sample to {save_path}")

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        manager.stop_system()