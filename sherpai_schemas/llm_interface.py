# Common OpenAI API interactions done by the SherpAI system

import json
import requests
import re
import ast

from typing import Any


def inference_conversation(
    system_prompt: str,
    user_prompt: str,
    model: str ,
    temperature: float = 0.0,
    base_url: str = "http://knowledgebase:8000",
    api_key: str = None,
) -> str:
    """Inference adjacent KnowledgeBase via OpenAI API interface.

    :param system_prompt: System prompt for the current task.
    :param user_prompt: User prompt for the current task.
    :param model: Name of LLM model or LoRA adapter.
    :param temperature: Creativity level of LLM
    :param base_url: URL of API endpoint
    :param api_key: Possible API key for authentication at external endpoint
    :"""

    # Use the adapter name if provided, otherwise the base model
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        full_url = base_url + "/v1/chat/completions"
        response = requests.post(full_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raises an error for 4xx or 5xx responses

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        return f"HTTP Request Error: {str(e)}"
    except KeyError:
        return f"Unexpected API Response Format: {response.text}"


def smart_cast(value: str, return_on_fail: Any) -> any:
    """Trun LLM response into python literals.

    :param value: LLM response
    :param return_on_fail: Default object when failed
    """
    if not isinstance(value, str):
        print(f"Warning: Input not string{value}")
        return value
    try:
        python_value = re.sub("true", "True", value)
        python_value = re.sub("false", "False", python_value)
        return ast.literal_eval(python_value)
    except (ValueError, SyntaxError):
        return return_on_fail if return_on_fail is not None else value
