# SherpAI-Schemas

Shared schema definitions for the SherpAI data quality pipeline.

## Installation

```bash
pip install git+https://github.com/SherpAI-Master/SherpAI-Schemas.git
```

## Schemas

### SherpAIInstance
Tracks identified data quality issues in a row, categorized by type: `incomplete`, `misplaced`, `formatting`, `misspelled`, `missing_value`, `validation`.

### Pair

### ToolUse

### Acceptance

## Usage

### Todo Example


## Requirements

- Python >= 3.12
- pandas >= 3.0.1

## Authors

Roman Klinghammer (rklinghammer@uni-potsdam.de)

def batch_inference_klassifik(remembered_names: pd.Series) -> pd.Series:
    """Batch inference all klassifik in a df."""
    prompts = [_format_gemma_prompt(Prompts.EXTRACT_KLASSIFIK_SYSTEM, str(name)) for name in remembered_names]
    results = inference_completion(model="unsloth/gemma-3-27b-it-bnb-4bit", prompt=prompts, max_tokens=60)
    choices = sorted(results["choices"], key=lambda x: x.get("index", 0))
    all_results = [choice["text"] for choice in choices]

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


def batch_inference_fix_formatting(remembered_formatting: pd.Series) -> pd.Series:
    """Führt Inferenz für alle Zeilen und Felder in einem einzigen Batch aus."""
    
    all_prompts = []
    structure_map = []

    for row_idx, row_list in remembered_formatting.items():
        for format_item in row_list:
            prompt = _format_gemma_prompt(Prompts.FIX_FORMATTING_SYSTEM, str(format_item[2]))
            all_prompts.append(prompt)
            structure_map.append((row_idx, format_item[1]))

    if not all_prompts:
        return pd.Series([SolutionInstance() for _ in range(len(remembered_formatting))], 
                         index=remembered_formatting.index)

    results = inference_completion(
        model="unsloth/gemma-3-27b-it-bnb-4bit", 
        prompt=all_prompts, 
        max_tokens=120
    )
    
    choices = sorted(results["choices"], key=lambda x: x.get("index", 0))
    all_texts = [choice["text"] for choice in choices]

    proposals_dict = {idx: SolutionInstance() for idx in remembered_formatting.index}

    for text, (row_idx, field_name) in zip(all_texts, structure_map):
        if not text:
            continue
            
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            useable_response = smart_cast(match.group(0), return_on_fail={})
            
            if useable_response and useable_response.get("fixable"):
                proposal = proposals_dict[row_idx]
                fix: Fix = getattr(proposal, field_name)
                fix.value = useable_response["data"]

    return pd.Series(proposals_dict.values(), index=remembered_formatting.index)


def batch_inference_fix_incomplete(remembered_incomplete: pd.Series) -> pd.Series:
    """Führt Inferenz für alle Zeilen und Felder in einem einzigen Batch aus."""
    
    all_prompts = []
    structure_map = []

    for row_idx, row_list in remembered_incomplete.items():
        for incomplete_item in row_list:
            prompt = _format_gemma_prompt(Prompts.FIX_INCOMPLETE_SYSTEM, str(incomplete_item[2]))
            all_prompts.append(prompt)
            structure_map.append((row_idx, incomplete_item[1]))

    print("All prompts", all_prompts)
    if not all_prompts:
        return pd.Series([SolutionInstance() for _ in range(len(remembered_incomplete))], 
                         index=remembered_incomplete.index)

    results = inference_completion(
        model="unsloth/gemma-3-27b-it-bnb-4bit", 
        prompt=all_prompts, 
        max_tokens=120
    )
    
    choices = sorted(results["choices"], key=lambda x: x.get("index", 0))
    all_texts = [choice["text"] for choice in choices]
    print("All incomplte correciton texts: ", all_texts)

    proposals_dict = {idx: SolutionInstance() for idx in remembered_incomplete.index}

    for text, (row_idx, field_name) in zip(all_texts, structure_map):
        if not text:
            continue
            
        match = re.search(r'"([^"]*)"', text, re.DOTALL)
        if match:
            useable_response = smart_cast(match.group(0), return_on_fail=None)
            
            if useable_response:
                proposal = proposals_dict[row_idx]
                fix: Fix = getattr(proposal, field_name)
                fix.value = useable_response

    return pd.Series(proposals_dict.values(), index=remembered_incomplete.index)