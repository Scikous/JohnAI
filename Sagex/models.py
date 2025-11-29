# import asyncio
# import gc
# import logging
# from abc import ABC, abstractmethod
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional
# import torch
# from model_utils import apply_chat_template, get_image


# # --- Configuration & Resource Objects ---
# @dataclass
# class LLMModelConfig:
#     """Configuration for loading an Exllamav2 model."""
#     main_model: str
#     tokenizer_model: str
#     revision: str = "8.0bpw"
#     is_vision_model: bool = True
#     max_seq_len: int = 65536
#     character_name: str = 'assistant'
#     instructions: str = ""

# @dataclass
# class Exllamav2Resources:
#     """A container for all loaded Exllamav2 resources."""
#     model: Any
#     cache: Any
#     exll2tokenizer: Any
#     generator: Any
#     gen_settings: Any
#     tokenizer: Any  # Transformers tokenizer
#     vision_model: Optional[Any] = None


import asyncio
import gc
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from model_utils import apply_chat_template, get_image

# --- vLLM specific imports ---
try:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    print("Warning: vLLM is not installed. The JohnVLLM class will not be available.")
    # Define dummy classes to avoid runtime errors if vLLM is not installed
    AsyncLLMEngine = object
    SamplingParams = object
    GuidedDecodingParams = object


# --- Configuration & Resource Objects ---
@dataclass
class LLMModelConfig:
    """Configuration for loading a model."""
    main_model: str
    tokenizer_model: str
    revision: str = "main"  # Use "main" as a more common default
    is_vision_model: bool = True
    max_seq_len: int = 65536
    character_name: str = 'assistant'
    instructions: str = ""
    # vLLM specific config
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90


@dataclass
class Exllamav2Resources:
    """A container for all loaded Exllamav2 resources."""
    model: Any
    cache: Any
    exll2tokenizer: Any
    generator: Any
    gen_settings: Any
    tokenizer: Any  # Transformers tokenizer
    vision_model: Optional[Any] = None


@dataclass
class VLLMResources:
    """A container for all loaded vLLM resources."""
    engine: AsyncLLMEngine
    tokenizer: Any  # Transformers tokenizer




# --- Base Class ---
class JohnLLMBase(ABC):
    """Abstract base class for John LLM models."""
    logger = logging.getLogger(__name__)  # Default logger -- stupid but easier than having to refactor a bunch

    def __init__(self, character_name: str, instructions: str, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.character_name = character_name
        self.instructions = instructions
        self.current_async_job = None

    @classmethod
    @abstractmethod
    async def load_model(cls, *args, **kwargs):
        """Loads all necessary model resources and returns an instance of the class."""
        pass

    @abstractmethod
    async def warmup(self):
        """
        Loads the model weights and performs any necessary warmup operations.
        This method should be called before the model is ready to generate text.
        """
        pass

    @abstractmethod
    async def dialogue_generator(self, prompt: str, **kwargs):
        """Generates a response to a prompt."""
        pass

    @abstractmethod
    async def cancel_dialogue_generation(self):
        """Requests cancellation of the ongoing dialogue generation."""
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleans up all resources held by the model."""
        pass

    # Async context manager for robust, async-aware cleanup
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False # Propagate exceptions


#######
class JohnExllamav2(JohnLLMBase):
    """
    Implementation of JohnLLMBase using the ExllamaV2 library.
    """
    def __init__(self, config: LLMModelConfig, resources: Exllamav2Resources, logger: logging.Logger = None):
        super().__init__(config.character_name, config.instructions, logger)
        self.config = config
        self.resources = resources

    @classmethod
    async def load_model(cls, config: LLMModelConfig):
        """
        Loads an ExLlamaV2 model based on the provided configuration.
        """
        from exllamav2 import (ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config,
                               ExLlamaV2Tokenizer, ExLlamaV2VisionTower)
        from exllamav2.generator import (ExLlamaV2DynamicGeneratorAsync, ExLlamaV2Sampler)
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        # 1. Load Transformers Tokenizer (for chat template)
        hf_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model)

        # 2. Load Exllamav2 Model Config
        try:
            exl2_config = ExLlamaV2Config(config.main_model)
        except:
            cls.logger.info(f"Local model not found. Downloading from Hugging Face Hub: {config.main_model}")
            hf_model_path = snapshot_download(repo_id=config.main_model, revision=config.revision)
            exl2_config = ExLlamaV2Config(hf_model_path)
        
        # 3. Load Vision Model (if applicable)
        vision_model = None
        if config.is_vision_model:
            cls.logger.info("Loading vision tower...")
            vision_model = ExLlamaV2VisionTower(exl2_config)
            vision_model.load(progress=True)

        # 4. Load Core Model & Tokenizer
        cls.logger.info("Loading main model...")
        model = ExLlamaV2(exl2_config)
        cache = ExLlamaV2Cache(model, max_seq_len=config.max_seq_len, lazy=True)
        model.load_autosplit(cache, progress=True)
        exll2_tokenizer = ExLlamaV2Tokenizer(exl2_config)
        
        # 5. Initialize Generator and Sampler Settings
        generator = ExLlamaV2DynamicGeneratorAsync(model=model, cache=cache, tokenizer=exll2_tokenizer)
        gen_settings = ExLlamaV2Sampler.Settings(
            temperature=1.8, top_p=0.95, min_p=0.08, top_k=50, token_repetition_penalty=1.05
        )

        # 6. Group resources and instantiate the class
        loaded_resources = Exllamav2Resources(
            model=model, cache=cache, exll2tokenizer=exll2_tokenizer, 
            generator=generator, gen_settings=gen_settings, 
            tokenizer=hf_tokenizer, vision_model=vision_model
        )

        instance = cls(config, loaded_resources)

        # 7. Perform warmup on the new instance
        # await instance.warmup()

        return instance

    async def _prepare_prompt_and_embeddings(self, prompt: str, assistant_prompt: Optional[str], conversation_history: Optional[List[str]]=None, images: Optional[List[Dict]]=None,
                                            add_generation_prompt: bool = True, continue_final_message: bool = False):
        """Handles image embedding and chat template application."""
        image_embeddings = None
        placeholders = ""

        if self.resources.vision_model and images:
            # Concurrently fetch and embed all images
            # image_tasks = images#[get_image(**img_args) for img_args in images]
            loaded_images = images#await asyncio.gather(*image_tasks)
            
            image_embeddings = [
                self.resources.vision_model.get_image_embeddings(
                    model=self.resources.model,
                    tokenizer=self.resources.exll2tokenizer,
                    image=img
                )
                for img in loaded_images
            ]
            placeholders = "\n".join([ie.text_alias for ie in image_embeddings]) + "\n"

        full_prompt = placeholders + prompt
        formatted_prompt = apply_chat_template(
            instructions=self.instructions,
            prompt=full_prompt,
            assistant_prompt=assistant_prompt,
            conversation_history=conversation_history,
            tokenizer=self.resources.tokenizer,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )
        input_ids = self.resources.exll2tokenizer.encode(
            formatted_prompt, add_bos=True, encode_special_tokens=True, embeddings=image_embeddings
        )
        
        return input_ids, image_embeddings

    async def dialogue_generator(self, prompt: str, assistant_prompt: Optional[str]=None, conversation_history: Optional[List[str]] = None, images: Optional[List[Dict]] = None, max_tokens: int = 200, add_generation_prompt: bool = True, continue_final_message: bool = False):
        """
        Generates character's response asynchronously.
        
        Args:
            prompt: Current user message string
            assistant_prompt: Optional prompt for the assistant to use as the base for it's response to complete it
            conversation_history: List of previous messages in order [user_msg1, assistant_msg1, user_msg2, assistant_msg2, ...]
            images: List of image dictionaries for vision model
            max_tokens: Maximum number of tokens to generate
            add_generation_prompt: Whether to add a generation prompt to the chat template
            continue_final_message: Whether to treat the prompt as a continuation of the assistant's message
        """
        from exllamav2.generator import ExLlamaV2DynamicJobAsync

        input_ids, image_embeddings = await self._prepare_prompt_and_embeddings(
            prompt, assistant_prompt, conversation_history, images, add_generation_prompt, continue_final_message
        )
        
        self.current_async_job = ExLlamaV2DynamicJobAsync(
            generator=self.resources.generator,
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            gen_settings=self.resources.gen_settings,
            completion_only=True,
            add_bos=False,
            stop_conditions=[self.resources.tokenizer.eos_token_id],
            embeddings=image_embeddings
        )
        return self.current_async_job

    async def cancel_dialogue_generation(self):
        """Cancels the currently ongoing ExLlamaV2DynamicJobAsync."""
        if self.current_async_job and hasattr(self.current_async_job, 'cancel'):
            self.logger.info("Cancelling current dialogue generation job.")
            try:
                await self.current_async_job.cancel()
            except Exception as e:
                self.logger.error(f"Error trying to cancel async_job: {e}")
        else:
            self.logger.warning("No cancellable dialogue generation job to cancel.")

    async def warmup(self):
        """
        Warms up the model by running a short, dummy generation.
        This pre-compiles CUDA kernels and initializes memory allocations,
        reducing latency on the first real generation request.
        """
        self.logger.info("Warming up the LLM... (This may take a moment)")
        try:
            # 1. Define a simple, short prompt. No need for history or images.
            warmup_prompt = "Hello, world."
            images = [
                # {"file": "media/test_image_1.jpg"},
                # {"file": "media/test_image_2.jpg"},
                # {"url": "https://media.istockphoto.com/id/1212540739/photo/mom-cat-with-kitten.jpg?s=612x612&w=0&k=20&c=RwoWm5-6iY0np7FuKWn8FTSieWxIoO917FF47LfcBKE="},
                {"url": "https://i.dailymail.co.uk/1s/2023/07/10/21/73050285-12283411-Which_way_should_I_go_One_lady_from_the_US_shared_this_incredibl-a-4_1689019614007.jpg"},
                {"url": "https://images.fineartamerica.com/images-medium-large-5/metal-household-objects-trevor-clifford-photography.jpg"}
            ]
            dummy_memory =[
                "Bob: The venerable oak tree, standing as an immutable sentinel through countless seasons, its gnarled branches reaching skyward like ancient, petrified arms, silently bore witness to the fleeting dramas of human endeavor unfolding beneath its rustling canopy, embodying a timeless wisdom far exceeding the ephemeral lifespan of any transient civilization.",
                "As the crimson sun dipped below the jagged horizon, casting long, ethereal shadows across the ancient, crumbling ruins, a lone figure, cloaked in worn, travel-stained fabric, paused to contemplate the vast, silent expanse of the desolate wasteland stretching endlessly before them, a chilling premonition of trials yet to come slowly solidifying in the depths of their weary soul.",
                "Bob: Scientists, meticulously analyzing the arcane data collected from the deepest recesses of the oceanic trenches, discovered astonishing, bioluminescent organisms exhibiting previously unknown adaptive mechanisms, providing tantalizing insights into the astonishing resilience of life in environments once deemed utterly inhospitable to any form of complex existence.",
                "Despite the overwhelming complexities",
                "Bob: and numerous unforeseen obstacles encountered during the arduous",
                "multi-year development cycle, the dedicated team of engineers, fueled by an unyielding passion for innovation and an unwavering commitment to their ambitious vision, ultimately managed to revolutionize the nascent field of quantum computing with their groundbreaking, paradigm-shifting invention.",
                "Bob: The venerable oak tree",
                "standing as an immutable sentinel through countless seasons",
                "Bob: its gnarled branches reaching skyward like ancient",
                "petrified arms",
                "Bob: silently bore witness to the fleeting dramas of human endeavor unfolding beneath its rustling canopy",
                "embodying a timeless wisdom far exceeding the ephemeral lifespan of any transient civilization.",
                "Bob: and numerous unforeseen obstacles encountered during the arduous",
                "multi-year development cycle, the dedicated team of engineers, fueled by an unyielding passion for innovation and an unwavering commitment to their ambitious vision, ultimately managed to revolutionize the nascent field of quantum computing with their groundbreaking, paradigm-shifting invention.",
                "Bob: Scientists, meticulously analyzing the arcane data collected from the deepest recesses of the oceanic trenches, discovered astonishing, bioluminescent organisms exhibiting previously unknown adaptive mechanisms, providing tantalizing insights into the astonishing resilience of life in environments once deemed utterly inhospitable to any form of complex existence.",
                "Despite the overwhelming complexities",
                "Bob: and numerous unforeseen obstacles encountered during the arduous",
                "embodying a timeless wisdom far exceeding the ephemeral lifespan of any transient civilization."
            ]


            warmup_job = await self.dialogue_generator(prompt=warmup_prompt, conversation_history=dummy_memory, images=images, max_tokens=512)

            # 4. Await the job's result to ensure it runs to completion.
            # We don't care about the output, just that the computation happens.
            async for _ in warmup_job:
                pass
                

            # 5. Clear the job reference to avoid confusion with real jobs.
            # The dialogue_generator method will set self.current_async_job later.
            self.current_async_job = None
            
            self.logger.info("LLM warmup complete. Model is ready.")

        except Exception as e:
            # Log an error but don't prevent the application from starting.
            self.logger.error(f"An error occurred during LLM warmup: {e}")

    async def cleanup(self):
        """Asynchronously cleans up model resources."""
        if self.current_async_job:
            self.logger.info("Attempting to cancel ongoing async_job during cleanup...")
            await self.cancel_dialogue_generation()
            self.current_async_job = None

        # Close the generator -- handles closing all pending tasks
        await self.resources.generator.close()
        # Release resources
        self.resources = None
        
        # Clear CUDA cache and run garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Cleaned up ExllamaV2 model resources.")



#######
# --- NEW VLLM IMPLEMENTATION ---
#######
class JohnVLLM(JohnLLMBase):
    """
    Implementation of JohnLLMBase using the vLLM library.
    """
    def __init__(self, config: LLMModelConfig, resources: VLLMResources, logger: logging.Logger = None):
        super().__init__(config.character_name, config.instructions, logger)
        self.config = config
        self.resources = resources
        # Use the job reference to store the vLLM request ID for cancellation
        self.current_async_job: Optional[str] = None

    @classmethod
    async def load_model(cls, config: LLMModelConfig):
        """
        Loads a vLLM model based on the provided configuration.
        """
        from transformers import AutoTokenizer

        cls.logger.info(f"Initializing vLLM engine for model: {config.main_model}")

        # 1. Configure and initialize the AsyncLLM engine
        engine_args = AsyncEngineArgs(
            model=config.main_model,
            revision=config.revision,
            tokenizer=config.tokenizer_model,
            trust_remote_code=True,
            max_model_len=config.max_seq_len,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            # For vision models, vLLM needs to know to load the image encoder
            enable_lora=config.is_vision_model,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        # 2. Load Transformers Tokenizer (for chat template)
        hf_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model)

        # 3. Group resources and instantiate the class
        loaded_resources = VLLMResources(engine=engine, tokenizer=hf_tokenizer)
        instance = cls(config, loaded_resources)
        
        return instance

    async def _prepare_prompt_and_data(self, prompt: str, assistant_prompt: Optional[str], conversation_history: Optional[List[str]]=None, images: Optional[List[Dict]]=None,
                                       add_generation_prompt: bool = True, continue_final_message: bool = False):
        """Handles image loading and chat template application for vLLM."""
        multi_modal_data = None
        placeholders = ""

        if self.config.is_vision_model and images:
            # vLLM expects <image> placeholders in the text and image data passed separately.
            placeholders = "<image>" * len(images) + "\n"
            
            # Concurrently fetch all images
            image_tasks = [get_image(**img_args) for img_args in images]
            loaded_images = await asyncio.gather(*image_tasks)
            multi_modal_data = {"image": loaded_images}

        full_prompt = placeholders + prompt
        formatted_prompt = apply_chat_template(
            instructions=self.instructions,
            prompt=full_prompt,
            assistant_prompt=assistant_prompt,
            conversation_history=conversation_history,
            tokenizer=self.resources.tokenizer,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )
        
        return formatted_prompt, multi_modal_data

    async def dialogue_generator(self, prompt: str, assistant_prompt: Optional[str]=None, conversation_history: Optional[List[str]] = None, images: Optional[List[Dict]] = None, max_tokens: int = 200, add_generation_prompt: bool = True, continue_final_message: bool = False, json_schema: Optional[Dict] = None):
        """
        Generates character's response asynchronously using vLLM's streaming engine.
        
        Args:
            ... (same as base class) ...
            json_schema: Optional Pydantic-style JSON schema for guided decoding.
        """
        formatted_prompt, multi_modal_data = await self._prepare_prompt_and_data(
            prompt, assistant_prompt, conversation_history, images, add_generation_prompt, continue_final_message
        )

        # Configure sampling parameters
        guided_params = None
        if json_schema:
            self.logger.info("Applying guided JSON decoding.")
            guided_params = GuidedDecodingParams(json_schema=json_schema)

        sampling_params = SamplingParams(
            temperature=0.8, top_p=0.95, max_tokens=max_tokens,
            stop_token_ids=[self.resources.tokenizer.eos_token_id],
            guided_decoding_params=guided_params,
        )

        # Generate a unique ID for this request
        request_id = f"john-llm-{uuid.uuid4()}"
        self.current_async_job = request_id

        # Get the async generator from the vLLM engine
        results_generator = self.resources.engine.generate(
            prompt=formatted_prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            multi_modal_data=multi_modal_data
        )

        # Stream the results
        try:
            async for request_output in results_generator:
                # In streaming, the new text is the difference from the previous output
                # For simplicity and robustness, we'll yield the full text of the first completion
                yield request_output.outputs[0].text
            
            # Final check to yield the complete text if anything was missed
            final_output: RequestOutput = request_output
            yield final_output.outputs[0].text

        except asyncio.CancelledError:
            self.logger.info(f"Dialogue generation for request {request_id} was cancelled.")
            # The cancellation is handled by the calling context, which should also call cancel_dialogue_generation
        finally:
            # Clean up the job reference once generation is complete or cancelled
            if self.current_async_job == request_id:
                self.current_async_job = None

    async def cancel_dialogue_generation(self):
        """Aborts the currently ongoing vLLM request."""
        if self.current_async_job:
            request_id = self.current_async_job
            self.logger.info(f"Cancelling current dialogue generation job: {request_id}")
            await self.resources.engine.abort(request_id)
            self.current_async_job = None # Clear immediately
        else:
            self.logger.warning("No active dialogue generation job to cancel.")

    async def warmup(self):
        """Warms up the vLLM engine by running a short, dummy generation."""
        self.logger.info("Warming up the vLLM engine... (This may take a moment)")
        try:
            warmup_prompt = "Hello, world. This is a test to warm up the model."
            images = None
            if self.config.is_vision_model:
                 images = [{"url": "https://images.fineartamerica.com/images-medium-large-5/metal-household-objects-trevor-clifford-photography.jpg"}]

            # Use the generator and iterate through it to completion
            warmup_stream = self.dialogue_generator(
                prompt=warmup_prompt,
                images=images,
                max_tokens=10
            )
            async for _ in warmup_stream:
                pass
            
            self.logger.info("vLLM engine warmup complete. Model is ready.")
        except Exception as e:
            self.logger.error(f"An error occurred during vLLM warmup: {e}", exc_info=True)

    async def cleanup(self):
        """Cleans up vLLM resources."""
        if self.current_async_job:
            self.logger.info("Attempting to cancel ongoing async_job during cleanup...")
            await self.cancel_dialogue_generation()

        # vLLM's engine doesn't have an explicit async shutdown method.
        # Aborting requests and releasing the reference is the primary cleanup.
        self.resources = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Cleaned up vLLM model resources.")