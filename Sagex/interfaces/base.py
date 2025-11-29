# llm_interface/base.py

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, AsyncGenerator

# --- Base Configuration ---
# @dataclass
# class BaseModelConfig:
#     """Base configuration for any LLM model."""
#     model_path_or_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"#"HuggingFaceTB/SmolLM2-135M-Instruct"
#     is_vision_model: bool = False
#     uses_special_chat_template: bool = False
#     max_seq_len: int = 4096
#     character_name: str = 'assistant'
#     instructions: str = ""
#     model_init_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BaseModelConfig:
    """
    Unified configuration for LLM models.
    Backend-specific settings (like gpu_memory_utilization for vLLM) 
    should go into model_init_kwargs.
    """
    model_path_or_id: str
    is_vision_model: bool = False
    uses_special_chat_template: bool = False
    # Path to tokenizer if different from model (common in ExLlama)
    tokenizer_path_or_id: Optional[str] = None 
    character_name: str = 'assistant'
    instructions: str = ""
    # Backend specific arguments (e.g. vllm's gpu_memory_utilization)
    model_init_kwargs: Dict[str, Any] = field(default_factory=dict)


# --- Synchronous Base Class ---
class JohnLLMBase(ABC):
    """Abstract base class for synchronous John LLM models."""

    def __init__(self, config: BaseModelConfig, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config
        self.character_name = config.character_name
        self.instructions = config.instructions

    @classmethod
    @abstractmethod
    def load_model(cls, config: BaseModelConfig) -> "JohnLLMBase":
        """Loads all necessary model resources and returns an instance of the class."""
        pass

    @abstractmethod
    def warmup(self):
        """Performs any necessary warmup operations."""
        pass

    @abstractmethod
    def dialogue_generator(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Generates a response to a prompt, yielding tokens as they are generated."""
        pass

    @abstractmethod
    def cancel_dialogue_generation(self):
        """Requests cancellation of the ongoing dialogue generation."""
        pass

    @abstractmethod
    def cleanup(self):
        """Cleans up all resources held by the model."""
        pass


class JohnLLMAsyncBase(ABC):
    """Abstract base class for asynchronous John LLM models."""

    def __init__(self, config: BaseModelConfig, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config
        self.character_name = config.character_name
        self.instructions = config.instructions

    @classmethod
    @abstractmethod
    async def load_model(cls, config: BaseModelConfig) -> "JohnLLMAsyncBase":
        """Asynchronously loads all necessary model resources."""
        pass

    @abstractmethod
    async def warmup(self):
        """Asynchronously performs any necessary warmup operations."""
        pass

    @abstractmethod
    async def dialogue_generator(
        self, 
        prompt: str, 
        assistant_prompt: Optional[str] = None,
        images: Optional[List[Dict]] = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        generation_config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generates text and yields token deltas (strings).
        Must accept `generation_config` for parameters like max_tokens, temperature, etc.
        """
        pass

    @abstractmethod
    async def cancel_dialogue_generation(self):
        """Asynchronously requests cancellation of the ongoing dialogue generation."""
        pass

    @abstractmethod
    async def cleanup(self):
        """Asynchronously cleans up all resources held by the model."""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False