#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#
# Derived from codespace-optimization/sfs (Apache-2.0); modified for CodeHelm.
#

from __future__ import annotations

import json
import logging
import os
from typing import Any

from .types import LLMResponse, Message

logger = logging.getLogger(__name__)


class ModelAdapter:
    """Wraps a CodeHelm CoderModel to provide the SFS generation interface."""

    def __init__(
        self,
        coder_model: Any,
        name: str = "codehelm_model",
        log_path: str = "",
    ) -> None:
        self.coder_model = coder_model
        self.name = name
        self.is_chat = True
        self.total_requests = 0
        self.total_tokens = 0
        self.output_tokens = 0
        self._log_path = log_path

    def reset_stats(self) -> None:
        """Reset token and request counters."""
        self.total_requests = 0
        self.total_tokens = 0
        self.output_tokens = 0

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count based on character length without requiring external tokenizers."""
        return max(1, len(text) // 4)

    def generate(
        self,
        input_data: list[Message] | str,
        max_tokens: int | None = None,
        temperature: float = 0.6,
        num_comps: int = 1,
        stop_strs: list[str] | None = None,
    ) -> list[LLMResponse]:
        """Generate completions from the underlying model."""
        results: list[LLMResponse] = []

        if isinstance(input_data, list):
            formatted_prompt = [
                {"role": m.role, "content": m.content}
                if hasattr(m, "role") and hasattr(m, "content")
                else m
                for m in input_data
            ]
            input_str = "\n".join(
                f"{getattr(m, 'role', '')}: {getattr(m, 'content', '')}"
                for m in input_data
            )
        else:
            formatted_prompt = input_data
            input_str = str(input_data)

        input_tokens = self._estimate_tokens(input_str)

        for _ in range(num_comps):
            kwargs: dict[str, Any] = {}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature

            try:
                raw_response = self.coder_model.generate(formatted_prompt, **kwargs)
            except TypeError:
                try:
                    raw_response = self.coder_model.generate(formatted_prompt)
                except Exception as exc:
                    logger.error("Model generation failed: %s", exc)
                    raise

            content = (
                raw_response if isinstance(raw_response, str) else str(raw_response)
            )
            output_tokens = self._estimate_tokens(content)

            self.total_requests += 1
            self.total_tokens += input_tokens + output_tokens
            self.output_tokens += output_tokens

            results.append(LLMResponse(content=content, logprob=0.0))

            if self._log_path:
                self._append_message_log(formatted_prompt, content, temperature)

        return results

    def _append_message_log(
        self,
        prompt: list[dict] | str,
        response: str,
        temperature: float,
    ) -> None:
        """Append one request/response record to the JSONL message log."""
        record = {
            "request_num": self.total_requests,
            "temperature": temperature,
            "messages": prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}],
            "response": response,
        }
        try:
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning("Could not write message log: %s", exc)
