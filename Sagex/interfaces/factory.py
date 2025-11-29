from typing import Dict, Any
from Sagex.interfaces.base import BaseModelConfig, JohnLLMAsyncBase

def create_llm_interface(
    backend: str,
    config_dict: Dict[str, Any],
    character_info: Dict[str, str]
) -> JohnLLMAsyncBase:
    """
    Factory function to instantiate the correct LLM interface.
    
    Args:
        backend: "vllm" or "exllamav2"
        config_dict: Dictionary from config.json containing 'llm_settings' or specific model args.
        character_info: tuple/dict containing (instructions, user_name, character_name)
    """
    
    # Construct generic config object
    # We map keys from typical config.json to BaseModelConfig
    model_init_kwargs = config_dict.get("model_init_kwargs", {})
    
    # Handle specific overrides if they exist in root of settings
    if "gpu_memory_utilization" in config_dict:
        model_init_kwargs["gpu_memory_utilization"] = config_dict["gpu_memory_utilization"]
    if "tensor_parallel_size" in config_dict:
        model_init_kwargs["tensor_parallel_size"] = config_dict["tensor_parallel_size"]

    model_config = BaseModelConfig(
        model_path_or_id=config_dict.get("llm_model_path"),
        tokenizer_path_or_id=config_dict.get("tokenizer_model_path", None),
        is_vision_model=config_dict.get("is_vision_model", False),
        character_name=character_info.get("character_name", "assistant"),
        instructions=character_info.get("instructions", ""),
        model_init_kwargs=model_init_kwargs
    )

    if backend == "vllm":
        from Sagex.interfaces.vllm_interface import JohnVLLMAsync
        return JohnVLLMAsync.load_model(model_config)
    elif backend == "exllamav2":
        from Sagex.interfaces.exllamav2_interface import JohnExllamav2
        return JohnExllamav2.load_model(model_config)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")