import gc
import logging
import uuid
from typing import Any, Dict, List, Optional, AsyncGenerator, Union

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams, RequestOutputKind#, StructuredOutputsParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from qwen_vl_utils import process_vision_info

from LLM_Wizard.utils.special_chat_templates import get_special_template_for_model

import torch
from model_utils import apply_chat_template

from LLM_Wizard.interfaces.base import BaseModelConfig, JohnLLMAsyncBase, JohnLLMBase


class _VLLMInterfaceMixin:
    """
    A mixin to share prompt preparation and sampling parameter logic for vLLM implementations.
    
    This class contains instance methods that depend on the state of the consumer
    class (e.g., tokenizer, instructions) and static methods for pure utility functions.
    """

    def _prepare_prompt(
        self,
        prompt: str,
        assistant_prompt: Optional[str] = None,
        conversation_history: Optional[List[str]] = None,
        images: Optional[List[Dict]] = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        **kwargs,
    ) -> str:        
        """Applies the chat template to build the final prompt string."""
        # This is an instance method and now correctly accesses self.tokenizer etc.
        return apply_chat_template(
            instructions=self.instructions,
            prompt=prompt,
            assistant_prompt=assistant_prompt,
            images=images,
            conversation_history=conversation_history,
            tokenizer=self.tokenizer,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )

    @staticmethod
    def _create_sampling_params(
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> SamplingParams:
        """
        Creates a vLLM SamplingParams object from a configuration dictionary.

        This helper function is responsible for parsing the generation configuration,
        specifically handling the dynamic creation of StructuredOutputsParams.

        Args:
            generation_config (Dict[str, Any], optional): A dictionary containing parameters
                for vLLM's SamplingParams, including a special 'guided_decoding' key.

        Returns:
            SamplingParams: A configured vLLM sampling parameters object.
        """
        config = generation_config.copy() if generation_config else {}

        guided_decoding_config = config.pop("guided_decoding", None)
        guided_decoding_params = None

        if guided_decoding_config:
            if isinstance(guided_decoding_config, dict) and len(guided_decoding_config) == 1:
                guided_decoding_params = StructuredOutputsParams(**guided_decoding_config)
            else:
                raise ValueError(
                    "'guided_decoding' in generation_config must be a dictionary "
                    "with a single key specifying the decoding type (e.g., 'json', 'regex')."
                )

        return SamplingParams(structured_outputs=guided_decoding_params, **config)

    def _prepare_inputs(
        self,
        prompt: str,
        assistant_prompt: Optional[str],
        conversation_history: Optional[List[str]],
        images: Optional[List[Dict]],
        add_generation_prompt: bool,
        continue_final_message: bool,
        generation_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Union[str, Dict, SamplingParams]]:
        """
        Prepares all common inputs required for the vLLM engine's generate call.
        
        This private helper centralizes the logic for prompt preparation and
        sampling parameter creation, serving both sync and async methods.
        """
        full_prompt, messages = self._prepare_prompt(
            prompt,
            assistant_prompt,
            conversation_history,
            images,
            add_generation_prompt,
            continue_final_message,
        )

        sampling_params = self._create_sampling_params(generation_config)

        print(messages)
        # Consolidate all inputs into a single dictionary
        if images and self._is_vision_model:
            image_inputs, _, video_kwargs, = process_vision_info(
                messages,
                image_patch_size=self.tokenizer.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True
            )
            return {
                "prompt": {"prompt": full_prompt, "multi_modal_data": {"image": image_inputs}, "mm_processor_kwargs": video_kwargs},
                "sampling_params": sampling_params,
            }
        
        return {"prompt": full_prompt, "sampling_params": sampling_params}


_DIALOGUE_GENERATOR_DOCSTRING = """
Generate text using the vLLM engine.

Args:
    prompt (str): The prompt to generate text from.
    assistant_prompt (str, optional): Optional prompt for the assistant to use as the base for its response.
    conversation_history (List, optional): List of previous messages in order [user_msg1, assistant_msg1, ...].
    images (List, optional): List of image dictionaries for vision models.
    add_generation_prompt (bool, optional): Whether to add the generation prompt. Defaults to True.
    continue_final_message (bool, optional): Whether to treat the prompt as a continuation of the assistant's message. Defaults to False.
    generation_config (Dict[str, Any], optional): A dictionary containing all sampling and generation parameters.
        This is used to create the `vllm.SamplingParams` object. For guided decoding, include a special
        'guided_decoding' key.

Example for `generation_config` with guided JSON:
generation_config = {{
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "guided_decoding": {{
        "json": '{{"type": "object", "properties": {{"name": {{"type": "string"}}}}}}'
    }}
}}

Returns:
{return_type}: {return_description}
"""

# --- Asynchronous vLLM Implementation ---

class JohnVLLMAsync(JohnLLMAsyncBase, _VLLMInterfaceMixin):
    """Asynchronous implementation of JohnLLMBase using vLLM."""

    def __init__(self, config: BaseModelConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.engine: Optional[AsyncLLM] = None
        self.tokenizer: Optional[Any] = None
        self.current_request_id: Optional[str] = None

    @classmethod
    async def load_model(cls, config: BaseModelConfig) -> "JohnVLLMAsync":
        instance = cls(config)
        instance.logger.info(f"Initializing vLLM AsyncLLM for model: {config.model_path_or_id}")
        
        engine_args_dict = {
            "model": config.model_path_or_id,
        }
        engine_args_dict.update(config.model_init_kwargs)
        
        engine_args = AsyncEngineArgs(**engine_args_dict)
        instance.engine = AsyncLLM.from_engine_args(engine_args)
        instance.tokenizer = await instance.engine.get_tokenizer()
        return instance

    @staticmethod
    def _get_output_kind(output_kind_str: str) -> RequestOutputKind:
        if not output_kind_str:
            return RequestOutputKind.CUMULATIVE
        if output_kind_str.upper() == "DELTA":
            return RequestOutputKind.DELTA

    async def dialogue_generator(self,
                             prompt: str,
                             assistant_prompt: Optional[str] = None,
                             conversation_history: Optional[List[str]] = None,
                             images: Optional[List[Dict]] = None,
                             add_generation_prompt: bool = True,
                             continue_final_message: bool = False,
                             generation_config: Optional[Dict[str, Any]] = None
                             ) -> AsyncGenerator[str, None]:
        #add the output kind DELTA to only receive the new text
        generation_config["output_kind"] = self._get_output_kind(generation_config["output_kind"])
        
        vllm_inputs = self._prepare_inputs(prompt, assistant_prompt, conversation_history, images,
            add_generation_prompt, continue_final_message, generation_config
        )
        self.current_request_id = f"john-llm-{uuid.uuid4().hex}"
        
        # vllm_inputs["sampling_params"]
        results_generator = self.engine.generate(
            vllm_inputs["prompt"], 
            sampling_params=vllm_inputs["sampling_params"], 
            request_id=self.current_request_id
        )
        
        async for request_output in results_generator:
            # vLLM yields cumulative text in outputs[0].text
            current_text = request_output.outputs[0].text
            yield current_text
        self.current_request_id = None

    async def cancel_dialogue_generation(self):
        if self.current_request_id:
            try:
                await self.engine.abort(self.current_request_id)
                self.logger.info(f"Aborted vLLM request: {self.current_request_id}")
            except Exception as e:
                self.logger.warning(f"Failed to abort vLLM request: {e}")
            self.current_request_id = None

    async def warmup(self):
        self.logger.info("Warming up the async vLLM engine...")
        try:
            gen_config = {"max_tokens": 10}
            async for _ in self.dialogue_generator(prompt="Hello", generation_config=gen_config):
                pass
            self.logger.info("Async vLLM engine warmup complete.")
        except Exception as e:
            self.logger.error(f"An error occurred during async vLLM warmup: {e}")

    async def cleanup(self):
        if self.current_request_id:
            await self.cancel_dialogue_generation()
        # Explicitly delete engine to free memory
        if self.engine:
            del self.engine
        self.engine = self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Cleaned up async vLLM resources.")

# --- Synchronous vLLM Implementation ---
class JohnVLLM(JohnLLMBase, _VLLMInterfaceMixin):
    """Synchronous implementation of JohnLLMBase using vLLM's LLM."""

    def __init__(self, config: BaseModelConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.engine: Optional[LLM] = None
        self.tokenizer: Optional[Any] = None
        self._is_vision_model: Optional[bool] = False

    @classmethod
    def load_model(cls, config: BaseModelConfig) -> "JohnVLLM":
        instance = cls(config)
        instance.logger.info(f"Initializing vLLM LLM for model: {config.model_path_or_id}")
        instance.engine = LLM(
            model=config.model_path_or_id,
            **config.model_init_kwargs
        )
        if config.is_vision_model:
            from transformers import AutoProcessor
            instance._is_vision_model = True
            instance.tokenizer = AutoProcessor.from_pretrained(config.model_path_or_id)
        else:
            instance.tokenizer = instance.engine.get_tokenizer()

        if config.uses_special_chat_template:
            instance.tokenizer.chat_template = get_special_template_for_model(config.model_path_or_id)
            instance.logger.info(f"Special chat template loaded for model: {config.model_path_or_id}")
        return instance

    def dialogue_generator(self,
                       prompt: str,
                       assistant_prompt: Optional[str] = None,
                       conversation_history: Optional[List[str]] = None,
                       images: Optional[List[Dict]] = None,
                       add_generation_prompt: bool = True,
                       continue_final_message: bool = False,
                       generation_config: Optional[Dict[str, Any]] = None
                       ) -> str:
        
        vllm_inputs = self._prepare_inputs(prompt, assistant_prompt, conversation_history, images,
            add_generation_prompt, continue_final_message, generation_config
        )

        outputs = self.engine.generate(vllm_inputs["prompt"], sampling_params=vllm_inputs["sampling_params"])
        return outputs[0].outputs[0].text

    # Set the docstring dynamically
    dialogue_generator.__doc__ = _DIALOGUE_GENERATOR_DOCSTRING.format(
        return_type="str",
        return_description="The complete generated text as a single string."
    )

    def cancel_dialogue_generation(self):
        self.logger.warning("Synchronous vLLM does not support cancellation of a running generator.")

    def warmup(self):
        self.logger.info("Warming up the sync vLLM engine...")
        try:
            # Consume the generator to ensure execution
            self.dialogue_generator(prompt="Hello")
            self.logger.info("Sync vLLM engine warmup complete.")
        except Exception as e:
            self.logger.error(f"An error occurred during sync vLLM warmup: {e}")

    def cleanup(self):
        del self.engine
        self.engine = self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Cleaned up sync vLLM resources.")