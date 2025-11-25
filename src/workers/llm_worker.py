import multiprocessing as mp
import os
import asyncio
from collections import deque
from typing import Dict, Any

from src.utils.env_utils import setup_project_root
from src.utils import logger as app_logger
from src.utils.performance_utils import apply_system_optimizations, async_check_gpu_memory
from src.common import config as app_config
from LLM_Wizard.model_utils import load_character, contains_sentence_terminator

# New Modular Interface Import
from LLM_Wizard.interfaces.factory import create_llm_interface

app_logger.setup_logging()
logger = app_logger.get_logger("LLMWorker")

async def llm_runner(shutdown_event, llm_control_queue, llm_to_tts_queue, gpu_ready_event, gpu_request_queue, worker_event, worker_id="LLM", llm_output_display_queue=None):
    config = app_config.load_config()
    apply_system_optimizations(logger, use_cuda=True)
    
    logger.info("LLM Worker: Loading model configuration...")
    llm_settings = config.get("llm_settings", {})
    
    # Load Character Info
    character_info_json_path = llm_settings.get("character_info_json")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    char_path = os.path.join(project_root, character_info_json_path) if not os.path.isabs(character_info_json_path) else character_info_json_path
    instructions, user_name, character_name = load_character(char_path)
    
    char_info_dict = {
        "instructions": instructions,
        "user_name": user_name,
        "character_name": character_name
    }

    # Instantiate Model via Factory
    # We default to 'vllm' if not specified, swapping out Exllama
    backend_type = llm_settings.get("backend", "vllm") 
    
    # Note: We await the factory call because load_model is async
    logger.info(f"LLM Worker: Initializing backend '{backend_type}'...")
    
    # Since the factory returns an awaitable for load_model, we handle it here.
    # But the factory function I wrote calls load_model (async) which returns the instance.
    # So create_llm_interface returns a Coroutine.
    llm_interface_coro = create_llm_interface(backend_type, llm_settings, char_info_dict)
    llm_model = await llm_interface_coro

    max_tokens = llm_settings.get("max_tokens", 512)
    conversation_history_maxlen = llm_settings.get("conversation_history_maxlen", 12)
    TERMINATE_OUTPUT = config.get("TERMINATE_OUTPUT", "w1zt3r")

    async def yield_and_queue_sentence(sentence: str):
        if not sentence:
            return
        logger.info(f"LLM Worker: Releasing GPU to queue sentence: '{sentence}'")
        gpu_request_queue.put({"type": "release", "worker_id": worker_id})
        try:
            while not shutdown_event.is_set():
                try:
                    llm_to_tts_queue.put(sentence, timeout=0.1)
                    logger.info(f"LLM Worker: Sentence queued successfully.")
                    return
                except mp.queues.Full:
                    await asyncio.sleep(0.05)
        finally:
            gpu_request_queue.put({"type": "acquire", "priority": 3, "worker_id":  worker_id})
            worker_event.wait()
            logger.info("LLM Worker: Re-acquiring GPU access.")

    # Context Manager for Cleanup
    async with llm_model:
        logger.info("✅ Main LLM model loaded and ready.")
        gpu_ready_event.set()
        
        conversation_history = deque(maxlen=conversation_history_maxlen)
        
        while not shutdown_event.is_set():
            try:
                if not llm_control_queue.empty():
                    control_message = llm_control_queue.get_nowait()
                    logger.info(control_message)
                    action = control_message.get("action")
                    
                    if action == "start":
                        continue_final_message = control_message.get("continue_final_message", False)
                        add_generation_prompt = control_message.get("add_generation_prompt", True)
                        stt_message = control_message.get("prompt")
                        
                        if conversation_history and continue_final_message:
                            prompt = conversation_history.pop()
                            conversation_history[-1] = stt_message
                        else:
                            prompt = stt_message
                            continue_final_message = False
                            add_generation_prompt = True

                        logger.info(f"LLM Worker: Starting generation for prompt: '{prompt}'")
                        
                        # Build Generation Config dynamically
                        generation_config = {
                            "max_tokens": llm_settings.get("max_tokens", 512),
                            "temperature": llm_settings.get("temperature", 0.7),
                            "top_p": llm_settings.get("top_p", 1.0),
                            "top_k": llm_settings.get("top_k", -1),
                            "repetition_penalty": llm_settings.get("repetition_penalty", 1.0),
                            "output_kind": llm_settings.get("output_kind", None),
                            # Guided decoding or other complex params can be added here
                        }
                        try:
                            gpu_request_queue.put({"type": "acquire", "priority": 2, "worker_id":  worker_id})
                            worker_event.wait()
                            await async_check_gpu_memory(logger)
                            
                            # Generic Generation Loop
                            # Supports both vLLM and Exllama via uniform interface
                            async_gen = llm_model.dialogue_generator(
                                prompt=prompt, 
                                conversation_history=list(conversation_history), 
                                generation_config=generation_config,
                                add_generation_prompt=add_generation_prompt, 
                                continue_final_message=continue_final_message
                            )

                            full_sentence = ""
                            full_response = ""
                            
                            async for token in async_gen:
                                if token:
                                    full_sentence += token
                                    if llm_output_display_queue:
                                        llm_output_display_queue.put(token)
                                else:
                                    continue
                                
                                if contains_sentence_terminator(full_sentence) and len(full_sentence) > 10:
                                    # Stripping logic
                                    sent_stripped = full_sentence.strip()
                                    logger.info(f"LLM Worker: Queueing sentence: '{sent_stripped}'")
                                    await yield_and_queue_sentence(sent_stripped)
                                    full_response += full_sentence
                                    full_sentence = ""
                                
                                # Check for Interrupts
                                if not llm_control_queue.empty():
                                    try:
                                        interrupt_msg = llm_control_queue.get_nowait()
                                        if interrupt_msg.get("action") == "interrupt":
                                            logger.warning("LLM Worker: Interrupt received.")
                                            # Flush queue logic
                                            while not llm_to_tts_queue.empty():
                                                llm_to_tts_queue.get_nowait()
                                            llm_to_tts_queue.put(TERMINATE_OUTPUT)
                                            
                                            await llm_model.cancel_dialogue_generation()
                                            full_response = "" 
                                            full_sentence = ""
                                            break
                                    except mp.queues.Empty:
                                        pass

                            # Handle any remaining text
                            if full_sentence.strip():
                                await yield_and_queue_sentence(full_sentence.strip())
                                full_response += full_sentence
                                
                            if full_response:
                                if not continue_final_message:
                                    conversation_history.append(prompt)
                                conversation_history.append(full_response)

                        finally:
                            logger.info("LLM Worker: Generation finished. Releasing GPU.")
                            gpu_request_queue.put({"type": "release", "worker_id": worker_id})
                            await asyncio.sleep(0.01)
                            
            except Exception as e:
                logger.error(f"Error in LLM runner loop: {e}", exc_info=True)
                await asyncio.sleep(1)

def llm_worker(shutdown_event: mp.Event, llm_control_queue: mp.Queue, llm_to_tts_queue: mp.Queue, gpu_ready_event: mp.Event, gpu_request_queue: mp.Queue, worker_event: mp.Event, llm_output_display_queue: mp.Queue):
    setup_project_root()
    worker_id = "LLM"
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(llm_runner(shutdown_event, llm_control_queue, llm_to_tts_queue, gpu_ready_event, gpu_request_queue, worker_event, worker_id, llm_output_display_queue))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        logger.info("LLM worker has shut down.")