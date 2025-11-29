minicpm_v_chat_template = """
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {%- set content = message['content'] -%}
        {%- if content is iterable and content is not string -%}
            {%- for part in content -%}
                {%- if part.type == "image" -%}
                    {{ '(<image>./</image>)' }}
                {%- endif -%}
            {%- endfor -%}
            {{ '\n' }}
            {%- for part in content -%}
                {%- if part.type == "text" -%}
                    {{ part.text }}
                {%- endif -%}
            {%- endfor -%}
        {%- else -%}
            {{ content }}
        {%- endif -%}
    {% elif message['role'] == 'assistant' %}
        {{ message['content'] }}
    {% endif %}
{%- endfor %}
"""








_SPECIAL_TEMPLATE_MAP = {
    "openbmb/MiniCPM-V-4_5": minicpm_v_chat_template,
    # "some-org/another-model-v1": another_model_template, # Future-proof
}


# --- Public API ---

def get_special_template_for_model(model_name_or_path: str) -> str | None:
    """
    Checks the registry for a special chat template for the given model.

    Args:
        model_name_or_path: The Hugging Face identifier for the model.

    Returns:
        The chat template string if a special one is required, otherwise None.
    """
    # You can make this logic as simple or complex as you need.
    # For now, we'll use an exact match.
    return _SPECIAL_TEMPLATE_MAP.get(model_name_or_path)