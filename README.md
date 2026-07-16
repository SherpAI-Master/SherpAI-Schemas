# SherpAI-Schemas

Shared schema definitions for the SherpAI data quality pipeline.

## Installation

```bash
pip install git+https://github.com/SherpAI-Master/SherpAI-Schemas.git
```

## Schemas

### SherpAIInstance
Tracks identified data quality issues in a row, as a list of `Finding`s.

### Finding
One detected problem in a row (`problem_type`: `incomplete`, `misplaced`, `formatting`, `misspelled`, `missing_value`, `validation`), plus its `detection` and, once proposed, `correction` -- both `Proposal`s.

### Proposal
A single tool's proposed detection or correction: what changed (`changes: list[FieldChange]`), and its review lifecycle (`LifecycleStatus`: drafted -> batching_ready/review_ready -> accepted/rejected).

### PipelineTool / PipelineRunner
Template Method execution skeleton for a single detection or correction tool: read input -> run `process_row` per row -> batch any pending LLM calls -> write output.

## Usage

### Todo Example


## Requirements

- Python >= 3.12
- pandas >= 3.0.1

## Authors

Roman Klinghammer (rklinghammer@uni-potsdam.de)