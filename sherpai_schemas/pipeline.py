"""Base classes for pipeline tools and the runner that drives them over a job's rows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .functions import parse_dimensions_from_str, parse_dimensions_to_str
from .llm_interface import format_gemma_prompt, inference_completion
from .schemas import (
    FieldChange,
    LifecycleStatus,
    Prompts,
    Proposal,
    SherpAIInstance,
    ToolIdentity,
)

DEFAULT_INPUT_PATH = Path("/job/input.jsonl")
DEFAULT_OUTPUT_PATH = Path("/job/output.jsonl")
DEFAULT_BATCH_MODEL = "unsloth/gemma-3-27b-it-bnb-4bit"


class PipelineTool:
    """Base class for a single detection/correction tool.

    Subclasses set `identity` and implement `process_row`. `batch_system_prompt`
    and `batch_max_tokens` are only relevant for correction tools that stage
    Proposals via `mark_batching_ready` for PipelineRunner to flush afterwards.
    """

    identity: ToolIdentity
    batch_system_prompt: Prompts | None = None
    batch_max_tokens: int = 60

    def process_row(self, row: pd.Series, instance: SherpAIInstance) -> SherpAIInstance:
        raise NotImplementedError


class PipelineRunner:
    """Runs a PipelineTool over every row of a job, then flushes any batched LLM work.

    Reads `input_path` (JSONL, one `SherpAISpace` column holding a stringified
    SherpAIInstance per row), applies any already-accepted corrections onto
    each row before calling `tool.process_row`, then writes `output_path`.
    """

    def __init__(
        self,
        tool: PipelineTool,
        input_path: Path = DEFAULT_INPUT_PATH,
        output_path: Path = DEFAULT_OUTPUT_PATH,
    ) -> None:
        self.tool = tool
        self.input_path = input_path
        self.output_path = output_path

    def run(self) -> None:
        df = pd.read_json(self.input_path, lines=True)
        if "SherpAISpace" not in df.columns:
            df["SherpAISpace"] = [SherpAIInstance() for _ in range(len(df))]
        else:
            df = parse_dimensions_from_str(df)

        for idx in df.index:
            row = df.loc[idx]
            instance: SherpAIInstance = row["SherpAISpace"]
            row = self._apply_accepted(row, instance)
            instance = self.tool.process_row(row, instance)
            df.loc[idx, row.index] = row
            df.at[idx, "SherpAISpace"] = instance

        self._flush_batches(df["SherpAISpace"])
        df = parse_dimensions_to_str(df)
        df.to_json(self.output_path, orient="records", lines=True)

    @staticmethod
    def _apply_accepted(row: pd.Series, instance: SherpAIInstance) -> pd.Series:
        """Write every accepted proposal's changes onto the row before processing it."""
        for finding in instance.findings:
            for proposal in (finding.detection, finding.correction):
                if proposal is None or proposal.status != LifecycleStatus.ACCEPTED:
                    continue
                for change in proposal.changes:
                    if change.column in row.index:
                        row[change.column] = change.value
        return row

    def _flush_batches(self, sherpai_col: pd.Series) -> None:
        """Send every BATCHING_READY proposal from this tool through one LLM call."""
        if self.tool.batch_system_prompt is None:
            return

        pending: list[Proposal] = [
            proposal
            for instance in sherpai_col
            for finding in instance.findings
            for proposal in (finding.detection, finding.correction)
            if proposal is not None
            and proposal.status == LifecycleStatus.BATCHING_READY
            and proposal.identity == self.tool.identity
        ]
        if not pending:
            return

        prompts = [
            format_gemma_prompt(self.tool.batch_system_prompt, proposal.pending_prompt)
            for proposal in pending
        ]
        results = inference_completion(
            model=DEFAULT_BATCH_MODEL, prompt=prompts, max_tokens=self.tool.batch_max_tokens
        )
        for proposal, result in zip(pending, results):
            for fix in result.fixes:
                proposal.changes = [FieldChange(column=fix.column, value=fix.corrected_value)]
                proposal.reason = fix.reason
            proposal.mark_review_ready()
