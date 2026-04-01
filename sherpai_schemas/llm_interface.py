# Common OpenAI API interactions done by the SherpAI system

import json
import requests
import pandas as pd


def inference_conversation(
    system_prompt: str,
    user_prompt: str,
    model: str,
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


def batch_vectorization(
    data: pd.Series,
    batch_size: int = 512,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    base_url: str = "http://knowledgebase:8000",
    api_key: str = None,
) -> list[float]:
    """Vectorize a series of strings with the provided model.

    :param data: Series with strings to be vectorized
    :type data: pd.Series
    :param batch_size: Size of batches which are vectorized, defaults to 512
    :type batch_size: int, optional
    :param embedding_model: defaults to "sentence-transformers/all-MiniLM-L6-v2"
    :type embedding_model: str, optional
    :return: Embedded strings
    :rtype: list[float]
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    embeddings = []
    for i in range(0, len(data), batch_size):
        batch_texts = data.iloc[i : i + batch_size].to_list()
        payload = {"model": model, "input": batch_texts}
        response = requests.post(
            base_url + "/v1/embeddings", json=payload, headers=headers, timeout=60
        )
        response.raise_for_status()
        data_json = response.json()
        batch_embeddings = [record["embedding"] for record in data_json["data"]]
        embeddings.extend(batch_embeddings)

    return embeddings
