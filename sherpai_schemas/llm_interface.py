# Common OpenAI API interactions done by the SherpAI system

import json
import requests
import re
import pandas as pd

from .schemas import SolutionInstance, Prompts
from .functions import smart_cast


def _format_gemma_prompt(system_prompt, user_prompt):
    return (f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n"
            f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n")


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


def inference_completion(
    prompt: str | list[str],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 50,
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
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        full_url = base_url + "/v1/completions"
        response = requests.post(full_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raises an error for 4xx or 5xx responses

        result = response.json()
        return result

    except requests.exceptions.RequestException as e:
        return f"HTTP Request Error: {str(e)}"
    except KeyError:
        return f"Unexpected API Response Format: {response.text}"


def batch_inference_klassifik(remembered_names: pd.Series) -> pd.Series:
    """Batch inference all klassifik in a df."""
    prompts = [_format_gemma_prompt(Prompts.EXTRACT_KLASSIFIK_SYSTEM, str(name)) for name in remembered_names]
    results = inference_completion(model="unsloth/gemma-3-27b-it-bnb-4bit", prompt=prompts, max_tokens=60)
    choices = sorted(results["choices"], key=lambda x: x.get("index", 0))
    all_results = [choice["text"] for choice in choices]
    print("EEEEEEEEE", all_results)

    obj_for_failed = {"prediction": 90, "reason": "Failed process!"}

    all_proposals = []

    for result in all_results:
        proposal = SolutionInstance()
        imputed_klassifik = obj_for_failed

        if result:
            match = re.search(r"\{.*\}", result, re.DOTALL)
            if match:
                imputed_klassifik = smart_cast(match.group(0), return_on_fail=obj_for_failed)
            else:
                print("No JSON object found in output")

        proposal.klassifik.value = imputed_klassifik["prediction"]
        proposal.klassifik.reason = imputed_klassifik["reason"]
        all_proposals.append(proposal)
    print("WWWWWWW", all_proposals)
    print("TTTTTTTT", pd.Series(all_proposals, index=remembered_names.index))
    print("12314", pd.Series(all_proposals, index=remembered_names.index)[0])
    print("12314", type(pd.Series(all_proposals, index=remembered_names.index)[0]))

    return pd.Series(all_proposals, index=remembered_names.index)


def batch_inference_address_extraction(remebered_snippet_lists: pd.Series) -> pd.Series:
    """Batch inference all address extracitons."""

    def _score_res_address(addr_list: list[dict]) -> int:
        """Evaluate completeness of extracted address by model."""
        best_addr = None, float("-inf")
        for addr in addr_list:
            score = 0
            if not addr:
                continue
            if re.match(r"^([A-Za-zÄÖÜäöüß])(?=.*\d).+", addr["street"]):
                score += 3
            if addr["city"] or len(addr["zip"]) == 5:
                score += 2
            if addr["country"]:
                score += 1
            if score > best_addr[1]:
                best_addr = addr, score
        return best_addr[0]

    all_prompts = []
    row_map = []

    # 1. Flatten everything into one big batch
    for row_idx, snippets in remebered_snippet_lists.items():
        for snip in snippets:
            all_prompts.append(_format_gemma_prompt(Prompts.EXTRACT_ADDRESS_SYSTEM, snip))
            row_map.append(row_idx)

    # 2. ONE API CALL for the whole DataFrame
    results = inference_completion(model="unsloth/gemma-3-27b-it-bnb-4bit", prompt=all_prompts, max_tokens=150)
    choices = sorted(results["choices"], key=lambda x: x.get("index", 0))
    all_results = [choice["text"] for choice in choices]

    # 3. Parse and group results by original row
    parsed_data = {} # {row_idx: [list_of_address_dicts]}
    for i, raw_text in enumerate(all_results):
        row_idx = row_map[i]
        if row_idx not in parsed_data: parsed_data[row_idx] = []
        
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        addr_obj = smart_cast(match.group(0), return_on_fail={}) if match else {}
        parsed_data[row_idx].append(addr_obj)

    # 4. Score and Build Proposals
    final_series_data = []
    for row_idx in remebered_snippet_lists.index:
        proposal = SolutionInstance()
        addresses = parsed_data.get(row_idx, [{}])
        best_res = _score_res_address(addresses)
        
        if best_res:
            proposal.zeile1.value = str(best_res.get("street", "")).replace(",", "_")
            proposal.ort.value = str(best_res.get("city", "")).replace(",", "_")
            proposal.plz.value = str(best_res.get("zip", "")).replace(",", "_")
            proposal.land.value = str(best_res.get("country", "")).replace(",", "_")
        else:
            proposal.zeile1.value = "LLM Error!"
            proposal.ort.value = "LLM Error!"
            proposal.land.value = "LLM Error!"
            proposal.plz.value = "LLM Error!"

        final_series_data.append(proposal)

    return pd.Series(final_series_data, index=remebered_snippet_lists.index)


def batch_vectorization(
    data: pd.Series,
    batch_size: int = 512,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    base_url: str = "http://embedbase:8000",
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
