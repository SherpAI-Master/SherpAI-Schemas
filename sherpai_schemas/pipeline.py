"""Template Method execution skeleton shared by every pipeline tool."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import pandas as pd

from .functions import parse_dimensions_from_str, parse_dimensions_to_str
from .llm_interface import format_gemma_prompt, inference_completion
from .schemas import (
    FieldChange,
    LifecycleStatus,
    LlmResponse,
    Prompts,
    Proposal,
    SherpAIInstance,
    ToolIdentity,
)


class PipelineTool(ABC):
    """One concrete tool, e.g. a single detection or correction step.

    Replaces the read/parse/apply/write skeleton that used to be duplicated
    verbatim across every images/*/main.py file.
    """

    identity: ClassVar[ToolIdentity]
    batch_system_prompt: ClassVar[Prompts | None] = None
    batch_max_tokens: ClassVar[int] = 60
    batch_model: ClassVar[str] = "unsloth/gemma-3-27b-it-bnb-4bit"

    @abstractmethod
    def process_row(self, row: pd.Series, instance: SherpAIInstance) -> SherpAIInstance:
        """Detect or correct issues in a single row, mutating and returning instance."""

    def apply_batch_result(self, proposal: Proposal, response: LlmResponse) -> None:
        """Write a batched LLM response back onto a BATCHING_READY proposal.

        Default applies each Fix 1:1 by column, then marks the proposal
        review-ready. Override for tools whose response shape needs different
        handling.
        """
        for fix in response.fixes:
            existing = next((change for change in proposal.changes if change.column == fix.column), None)
            if existing is not None:
                existing.value = fix.corrected_value
            else:
                proposal.changes.append(FieldChange(column=fix.column, value=fix.corrected_value))
            proposal.reason = fix.reason

        proposal.mark_review_ready()


class PipelineRunner:
    """Template Method: read -> per-row -> flush_batches -> write."""

    def __init__(
        self,
        tool: PipelineTool,
        input_path: Path = Path("/job/input.jsonl"),
        output_path: Path = Path("/job/output.jsonl"),
    ) -> None:
        self.tool = tool
        self.input_path = input_path
        self.output_path = output_path

    def run(self) -> None:
        df = self._read()
        df = self._process_rows(df)
        df = self._flush_batches(df)
        self._write(df)

    def _read(self) -> pd.DataFrame:
        df = pd.read_json(self.input_path, lines=True)
        return parse_dimensions_from_str(df)

    def _process_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        def _run_tool(row: pd.Series) -> SherpAIInstance:
            instance: SherpAIInstance = row["SherpAISpace"]
            # Bake in corrections already accepted by earlier tools before this
            # tool sees the row, so every tool always works off the latest data.
            row = instance.apply_solutions(row)
            return self.tool.process_row(row, instance)

        df["SherpAISpace"] = df.apply(_run_tool, axis=1)
        return df

    def _flush_batches(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.tool.batch_system_prompt is None:
            return df

        pending: list[Proposal] = []
        for instance in df["SherpAISpace"]:
            if not isinstance(instance, SherpAIInstance):
                continue
            for finding in instance.findings:
                for proposal in (finding.detection, finding.correction):
                    if proposal is not None and proposal.status == LifecycleStatus.BATCHING_READY:
                        pending.append(proposal)

        if not pending:
            return df

        prompts = [
            format_gemma_prompt(self.tool.batch_system_prompt, proposal.pending_prompt)
            for proposal in pending
        ]
        results = inference_completion(
            model=self.tool.batch_model, prompt=prompts, max_tokens=self.tool.batch_max_tokens
        )
        if len(results) != len(pending):
            msg = "Mismatch between number of prompts sent and results received"
            raise ValueError(msg)

        for proposal, response in zip(pending, results):
            self.tool.apply_batch_result(proposal, response)

        return df

    def _write(self, df: pd.DataFrame) -> None:
        df = parse_dimensions_to_str(df)
        df.to_json(self.output_path, lines=True, orient="records")
